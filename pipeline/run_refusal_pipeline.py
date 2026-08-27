"""End-to-end pipeline for one concept, tonight run on refusal / Qwen3-8B.

contrastive pairs -> probe -> diff-in-means direction -> steer -> judge readout

LAYER is hardcoded rather than picked by argmax over a per-layer scan: the
first pilot's scan showed a plateau of AUROC 1.000 across layers 13-24 (real
concept separation, not the lexical-leakage artifact we first suspected --
extraction is prompt-only, no completion tokens are ever present at that
point), and argmax breaks ties by picking the first one. Layer 18 sits at the
middle of that plateau, ~50% depth, matching where the refusal-direction
literature typically finds it, and is a call made before looking at any
further data rather than a data-driven selection step that would need
defending. The AUROC-by-layer curve is still computed and reported below, but
purely descriptively -- LAYER does not depend on it.

The judge is external (api_judge.py, Claude Haiku via the Anthropic API), not
the model's own logits. Two self-judge attempts were built and both were
inverted or uninformative (see judge.py's history) -- an external judge
removes the confound of the model scoring its own output at a layer we're
actively perturbing. Gate 1 (api_judge.check_judge_separation) runs before
the model is even loaded: it's two cheap API calls, and if it fails the
problem is the API key or prompt parsing, not the research, so there's no
reason to spend GPU time first. Gate 2 (api_judge.validate_against_keyword)
runs after the sweep: the judge is only trusted if it clearly agrees with the
keyword detector on which completions are refusals.

Alpha is capped to [-0.5 .. 0.5] in multiples of the mean residual norm at
LAYER -- an earlier +-1.0 sweep produced incoherent/degenerate completions,
which is not a measurement of steering strength, just of broken generations.

Timing is tracked per-unit (per activation extraction, per generation, per
judge call) and extrapolated to the full Saturday config, since tonight's toy
run is far smaller than the real one and its raw wall-clock would pass the
45-minute gate vacuously.
"""

import json
import time
from pathlib import Path

import torch

from .api_judge import check_judge_separation, score_batch, validate_against_keyword
from .direction import diff_in_means
from .extract import extract_layer_activations
from .judge import is_refusal
from .model import load_model
from .pairs import get_pairs
from .probe import fit_final_probe, scan_layers, shuffle_label_control
from .steer import generate_steered

CONCEPT = "refusal"
LAYER = 18
N_HOLDOUT = 12
ALPHA_MULTIPLIERS = [-0.5, -0.35, -0.2, -0.1, 0.0, 0.1, 0.2, 0.35, 0.5]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Saturday's real scale (per user estimate): ~300 activation extractions and
# ~400 generations per concept, across 8 concepts.
FULL_RUN_CONCEPTS = 8
FULL_RUN_ACTIVATIONS_PER_CONCEPT = 300
FULL_RUN_GENERATIONS_PER_CONCEPT = 400


def main():
    timings = {}

    def timed(name, fn, *args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start)
        return result

    gate1_refusal_score, gate1_comply_score = check_judge_separation(CONCEPT)
    print(f"Gate 1 (judge sanity, no GPU yet): hard-refusal={gate1_refusal_score} "
          f"full-compliance={gate1_comply_score}", flush=True)
    gate1_ok = (
        gate1_refusal_score is not None and gate1_comply_score is not None
        and gate1_refusal_score - gate1_comply_score > 0.5
    )
    if not gate1_ok:
        print("Gate 1 FAILED -- judge does not separate hand-written extremes. "
              "Stopping before loading the model: this is an API key/parsing "
              "problem, not a research result.", flush=True)
        return
    print("Gate 1 passed.", flush=True)

    model, tokenizer = timed("load_model", load_model)
    print("model loaded", flush=True)

    pairs = get_pairs(CONCEPT)
    holdout_pairs = pairs[-N_HOLDOUT:]
    fit_pairs = pairs[:-N_HOLDOUT]

    n_extractions = 0

    def extract(prompts, label):
        nonlocal n_extractions
        n_extractions += len(prompts)
        result = timed("extract_activations", extract_layer_activations, model, tokenizer, prompts)
        print(f"extracted {label} ({len(prompts)} prompts, {n_extractions} total so far)", flush=True)
        return result

    fit_pos = extract([p.positive for p in fit_pairs], "fit_pos")
    fit_neg = extract([p.negative for p in fit_pairs], "fit_neg")

    scan_results = timed("probe_scan", scan_layers, fit_pos, fit_neg)
    auroc_by_layer = {l: round(scan_results[l]["auroc"], 4) for l in sorted(scan_results)}
    print(f"auroc_by_layer (descriptive only, LAYER={LAYER} is hardcoded): {auroc_by_layer}", flush=True)

    probe, held_out_auroc = timed("final_probe_fit", fit_final_probe, fit_pos, fit_neg, LAYER)
    shuffled_auroc = timed("shuffle_control", shuffle_label_control, fit_pos, fit_neg, LAYER)
    print(f"held_out_auroc={held_out_auroc:.3f} shuffle_control_auroc={shuffled_auroc:.3f}", flush=True)

    direction = diff_in_means(fit_pos, fit_neg, LAYER)
    layer_acts = torch.cat([fit_pos[LAYER], fit_neg[LAYER]])
    mean_residual_norm = layer_acts.norm(dim=-1).mean().item()
    print(f"mean_residual_norm at layer {LAYER}: {mean_residual_norm:.1f}", flush=True)

    n_generations = 0
    all_completions, all_keyword_flags, per_alpha_slices = [], [], []
    for multiplier in ALPHA_MULTIPLIERS:
        alpha = multiplier * mean_residual_norm
        start_idx = len(all_completions)
        for i, pair in enumerate(holdout_pairs):
            completion = timed(
                "steer_generate", generate_steered,
                model, tokenizer, pair.positive, LAYER, direction, alpha,
            )
            n_generations += 1
            keyword_refusal = is_refusal(completion)
            all_completions.append(completion)
            all_keyword_flags.append(keyword_refusal)
            print(f"multiplier={multiplier} alpha={alpha:.1f} [{i + 1}/{len(holdout_pairs)}] "
                  f"keyword_refusal={keyword_refusal}", flush=True)
        per_alpha_slices.append((multiplier, alpha, slice(start_idx, len(all_completions))))

    print(f"scoring {len(all_completions)} completions with the external judge...", flush=True)
    graded_scores = timed("api_judge_scoring", score_batch, all_completions, CONCEPT)

    sweep_results = []
    for multiplier, alpha, sl in per_alpha_slices:
        keyword_flags = all_keyword_flags[sl]
        scores = [s for s in graded_scores[sl] if s is not None]
        examples = [
            {"prompt": pair.positive, "completion": completion,
             "keyword_refusal": kw, "graded_refusal_score": score}
            for pair, completion, kw, score in zip(
                holdout_pairs, all_completions[sl], keyword_flags, graded_scores[sl],
            )
        ]
        agreement = sum(
            kw == (score >= 0.5) for kw, score in zip(keyword_flags, graded_scores[sl]) if score is not None
        ) / len(scores) if scores else float("nan")
        sweep_results.append({
            "alpha_multiplier": multiplier,
            "alpha": alpha,
            "keyword_refusal_rate": sum(keyword_flags) / len(keyword_flags),
            "mean_graded_refusal_score": sum(scores) / len(scores) if scores else None,
            "judge_agreement_rate": agreement,
            "examples": examples,
        })
        print(f"multiplier={multiplier} done: keyword_refusal_rate={sweep_results[-1]['keyword_refusal_rate']:.2f} "
              f"mean_graded_refusal_score={sweep_results[-1]['mean_graded_refusal_score']}", flush=True)

    print("Gate 2 (judge vs. keyword anchor):", flush=True)
    gate2_gap = validate_against_keyword(all_completions, all_keyword_flags, CONCEPT)
    gate2_ok = gate2_gap is not None and gate2_gap > 0.30

    time_per_extraction = timings["extract_activations"] / n_extractions
    time_per_generation = timings["steer_generate"] / n_generations
    time_per_judge_call = timings["api_judge_scoring"] / n_generations
    predicted_full_run_hours = FULL_RUN_CONCEPTS * (
        FULL_RUN_ACTIVATIONS_PER_CONCEPT * time_per_extraction
        + FULL_RUN_GENERATIONS_PER_CONCEPT * (time_per_generation + time_per_judge_call)
    ) / 3600

    total = sum(timings.values())
    print(f"tonight's total: {total:.1f}s over {n_extractions} extractions, {n_generations} generations")
    print(f"layer={LAYER} held_out_auroc={held_out_auroc:.3f} shuffle_control_auroc={shuffled_auroc:.3f}")
    print(f"Gate 2 gap={gate2_gap:+.2f} {'PASS' if gate2_ok else 'FAIL'}")
    print(f"predicted full Saturday run ({FULL_RUN_CONCEPTS} concepts, "
          f"{FULL_RUN_ACTIVATIONS_PER_CONCEPT} extractions + {FULL_RUN_GENERATIONS_PER_CONCEPT} "
          f"generations each): {predicted_full_run_hours:.2f}h")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{CONCEPT}_run.json"
    out_path.write_text(json.dumps({
        "concept": CONCEPT,
        "gate1_judge_sanity_check": {
            "hard_refusal_score": gate1_refusal_score,
            "full_compliance_score": gate1_comply_score,
            "passed": gate1_ok,
        },
        "gate2_keyword_agreement_gap": gate2_gap,
        "gate2_passed": gate2_ok,
        "layer": LAYER,
        "auroc_by_layer": auroc_by_layer,
        "held_out_auroc": held_out_auroc,
        "shuffle_control_auroc": shuffled_auroc,
        "mean_residual_norm": mean_residual_norm,
        "timings_sec": timings,
        "total_sec": total,
        "n_extractions": n_extractions,
        "n_generations": n_generations,
        "time_per_extraction_sec": time_per_extraction,
        "time_per_generation_sec": time_per_generation,
        "time_per_judge_call_sec": time_per_judge_call,
        "predicted_full_run_hours": predicted_full_run_hours,
        "sweep": sweep_results,
    }, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
