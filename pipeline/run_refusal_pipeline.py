"""End-to-end pipeline for one concept, tonight run on refusal / Qwen3-8B.

contrastive pairs -> probe (layer scan) -> diff-in-means direction -> steer -> judge readout

Layer selection and the reported probe AUROC are fit on disjoint splits (see
probe.scan_layers docstring) and a label-shuffle control is run alongside the
real probe, so a passing AUROC actually means something instead of just
reflecting d >> n. Timing is tracked per-unit (per activation extraction, per
generation, per judge call) and extrapolated to the full Saturday config,
since tonight's toy run (60 pairs, 9 alphas x 12 holdout prompts) is far
smaller than the real one and its raw wall-clock would pass the 45-minute
gate vacuously.

Thinking mode is disabled everywhere via model.encode_chat (Qwen3 defaults to
emitting a <think> block that can eat the whole generation budget before any
actual answer appears -- with it on, the judge scores deliberation, not a
response). Alpha is swept symmetrically and expressed as a multiple of the
mean residual norm at the chosen layer, since a fixed raw alpha can't be
compared across layers/models with very different activation scales. A judge
sanity check runs first: if a hand-written hard refusal and a hand-written
full compliance don't score clearly apart, the judge is outputting a prior,
not reading the text, and the rest of the run is diagnostic noise regardless
of what the sweep shows.
"""

import json
import time
from pathlib import Path

import torch

from .direction import diff_in_means
from .extract import extract_layer_activations
from .judge import check_judge_separation, digit_token_ids, graded_refusal_score, is_refusal
from .model import load_model
from .pairs import get_pairs
from .probe import best_layer, fit_final_probe, scan_layers, shuffle_label_control
from .steer import generate_steered

CONCEPT = "refusal"
N_HOLDOUT = 12
ALPHA_MULTIPLIERS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
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

    model, tokenizer = timed("load_model", load_model)
    print("model loaded", flush=True)

    digit_ids = digit_token_ids(tokenizer)
    sanity_refusal_score, sanity_comply_score = check_judge_separation(model, tokenizer, digit_ids)
    print(f"judge sanity check: hard-refusal={sanity_refusal_score:.2f} "
          f"full-compliance={sanity_comply_score:.2f} "
          f"(want these clearly apart, e.g. >0.3 gap)", flush=True)
    if abs(sanity_refusal_score - sanity_comply_score) < 0.3:
        print("WARNING: judge does not separate hand-written extremes -- "
              "treat every downstream score below as unvalidated", flush=True)

    pairs = get_pairs(CONCEPT)
    holdout_pairs = pairs[-N_HOLDOUT:]
    remainder = pairs[:-N_HOLDOUT]
    half = len(remainder) // 2
    select_pairs, eval_pairs = remainder[:half], remainder[half:]

    n_extractions = 0

    def extract(prompts, label):
        nonlocal n_extractions
        n_extractions += len(prompts)
        result = timed("extract_activations", extract_layer_activations, model, tokenizer, prompts)
        print(f"extracted {label} ({len(prompts)} prompts, {n_extractions} total so far)", flush=True)
        return result

    select_pos = extract([p.positive for p in select_pairs], "select_pos")
    select_neg = extract([p.negative for p in select_pairs], "select_neg")
    scan_results = timed("probe_scan", scan_layers, select_pos, select_neg)
    layer = best_layer(scan_results)
    selection_auroc = scan_results[layer]["auroc"]
    auroc_by_layer = {l: round(scan_results[l]["auroc"], 4) for l in sorted(scan_results)}
    print(f"layer scan done: layer={layer} selection_auroc={selection_auroc:.3f}", flush=True)
    print(f"auroc_by_layer: {auroc_by_layer}", flush=True)

    eval_pos = extract([p.positive for p in eval_pairs], "eval_pos")
    eval_neg = extract([p.negative for p in eval_pairs], "eval_neg")
    probe, held_out_auroc = timed("final_probe_fit", fit_final_probe, eval_pos, eval_neg, layer)
    shuffled_auroc = timed("shuffle_control", shuffle_label_control, eval_pos, eval_neg, layer)
    print(f"held_out_auroc={held_out_auroc:.3f} shuffle_control_auroc={shuffled_auroc:.3f}", flush=True)

    direction = diff_in_means(eval_pos, eval_neg, layer)
    layer_acts = torch.cat([eval_pos[layer], eval_neg[layer]])
    mean_residual_norm = layer_acts.norm(dim=-1).mean().item()
    print(f"mean_residual_norm at layer {layer}: {mean_residual_norm:.1f}", flush=True)

    n_generations = 0
    sweep_results = []
    for multiplier in ALPHA_MULTIPLIERS:
        alpha = multiplier * mean_residual_norm
        keyword_flags, graded_scores, examples = [], [], []
        for i, pair in enumerate(holdout_pairs):
            completion = timed(
                "steer_generate", generate_steered,
                model, tokenizer, pair.positive, layer, direction, alpha,
            )
            n_generations += 1
            keyword_refusal = is_refusal(completion)
            graded_score = timed(
                "judge_score", graded_refusal_score,
                model, tokenizer, pair.positive, completion, digit_ids,
            )
            keyword_flags.append(keyword_refusal)
            graded_scores.append(graded_score)
            examples.append({
                "prompt": pair.positive,
                "completion": completion,
                "keyword_refusal": keyword_refusal,
                "graded_refusal_score": graded_score,
            })
            print(f"multiplier={multiplier} alpha={alpha:.1f} [{i + 1}/{len(holdout_pairs)}] "
                  f"keyword_refusal={keyword_refusal} graded={graded_score:.2f}", flush=True)

        agreement = sum(
            kw == (score >= 0.5) for kw, score in zip(keyword_flags, graded_scores)
        ) / len(keyword_flags)
        sweep_results.append({
            "alpha_multiplier": multiplier,
            "alpha": alpha,
            "keyword_refusal_rate": sum(keyword_flags) / len(keyword_flags),
            "mean_graded_refusal_score": sum(graded_scores) / len(graded_scores),
            "judge_agreement_rate": agreement,
            "examples": examples,
        })
        print(f"multiplier={multiplier} done: keyword_refusal_rate={sweep_results[-1]['keyword_refusal_rate']:.2f} "
              f"mean_graded_refusal_score={sweep_results[-1]['mean_graded_refusal_score']:.2f}", flush=True)

    time_per_extraction = timings["extract_activations"] / n_extractions
    time_per_generation = timings["steer_generate"] / n_generations
    time_per_judge_call = timings["judge_score"] / n_generations
    predicted_full_run_hours = FULL_RUN_CONCEPTS * (
        FULL_RUN_ACTIVATIONS_PER_CONCEPT * time_per_extraction
        + FULL_RUN_GENERATIONS_PER_CONCEPT * (time_per_generation + time_per_judge_call)
    ) / 3600

    total = sum(timings.values())
    print(f"tonight's total: {total:.1f}s over {n_extractions} extractions, {n_generations} generations")
    print(f"layer={layer} selection_auroc={selection_auroc:.3f} "
          f"held_out_auroc={held_out_auroc:.3f} shuffle_control_auroc={shuffled_auroc:.3f}")
    print(f"predicted full Saturday run ({FULL_RUN_CONCEPTS} concepts, "
          f"{FULL_RUN_ACTIVATIONS_PER_CONCEPT} extractions + {FULL_RUN_GENERATIONS_PER_CONCEPT} "
          f"generations each): {predicted_full_run_hours:.2f}h")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{CONCEPT}_run.json"
    out_path.write_text(json.dumps({
        "concept": CONCEPT,
        "judge_sanity_check": {
            "hard_refusal_score": sanity_refusal_score,
            "full_compliance_score": sanity_comply_score,
        },
        "layer": layer,
        "selection_auroc": selection_auroc,
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
