"""Generalized pipeline: loops over concepts, each doing
extract -> probe (fixed layer) -> diff-in-means -> alpha sweep -> judge ->
unsteered probe-response readout on the generated completions.

LAYER = 18 is fixed for every concept, not re-tuned per concept: if a concept
probes poorly there, that's a finding about the concept, not a reason to
chase a better layer for it.

All three concepts extract on full (question, answer) turns, not bare
prompts -- a deployed monitor reads generated text, so training the probe on
prompt-only activations (predicting refusal before the model answers) would
measure a different, easier question than the one this project is about, and
would bias the probe toward looking artificially hard to evade on generated
completions later (it was barely trained on completion-bearing activations
in the first place). refusal's fit pairs (see pairs.py) are two different
*requests*, not one question with two candidate answers, so its extraction
pairs each request with a canned completion matching the expected stance,
randomly assigned per pair from several varied phrasings (pairs.py's
REFUSAL_COMPLETIONS / COMPLIANCE_COMPLETIONS) rather than a bespoke one per
request. A single fixed completion per class was tried first and is wrong --
see the comment above those lists for why. deception/non_english's fit pairs
already carry real candidate answers per question. (extract.
extract_layer_activations, prompt-only, is kept available for a Monday
red-team check on extraction basis, but no concept currently uses it.)

Evasion cost is the new thing this run adds: for every generated completion
we now also read the probe's response, unsteered, on the actual
(prompt, completion) text -- since a deployed monitor sees the text, not our
steering vector, this has to be a clean forward pass with no hook attached.
Plotting probe_response against the judge's behavioural score per alpha is
the evasion-cost scatter: a concept where behaviour swings a lot per unit of
probe movement is fragile, one where they move together is robust.

Gate 1 (api_judge.check_judge_separation) runs once, before the model loads.
Gate 2 (api_judge.validate_against_keyword) only applies to refusal, since
it's the only concept with a working keyword detector.
"""

import json
import random
import time
from pathlib import Path

import torch

from .api_judge import check_judge_separation, score_batch, validate_against_keyword
from .direction import diff_in_means
from .extract import extract_layer_activations_for_answers
from .judge import is_refusal
from .model import load_model
from .pairs import COMPLIANCE_COMPLETIONS, CONCEPTS, REFUSAL_COMPLETIONS, AnswerPair
from .probe import fit_final_probe, probe_response, scan_layers, shuffle_label_control
from .steer import generate_steered

LAYER = 18
ALPHA_MULTIPLIERS = [-0.5, -0.35, -0.2, -0.1, 0.0, 0.1, 0.2, 0.35, 0.5]
CONCEPTS_TO_RUN = ["refusal", "deception", "non_english"]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Fixed seed: which canned phrasing lands on which refusal pair must be
# reproducible across runs, not re-rolled every time the pipeline executes.
_CANNED_COMPLETION_SEED = 0


def _extract_fit_activations(model, tokenizer, spec, timed):
    fit_pairs = spec["fit_pairs"]
    if isinstance(fit_pairs[0], AnswerPair):
        pos_examples = [(p.question, p.positive_answer) for p in fit_pairs]
        neg_examples = [(p.question, p.negative_answer) for p in fit_pairs]
    else:  # refusal's Pair: two different requests, varied canned completions
        rng = random.Random(_CANNED_COMPLETION_SEED)
        pos_examples = [(p.positive, rng.choice(REFUSAL_COMPLETIONS)) for p in fit_pairs]
        neg_examples = [(p.negative, rng.choice(COMPLIANCE_COMPLETIONS)) for p in fit_pairs]
    pos = timed("extract", extract_layer_activations_for_answers, model, tokenizer, pos_examples)
    neg = timed("extract", extract_layer_activations_for_answers, model, tokenizer, neg_examples)
    return pos, neg, len(fit_pairs) * 2


def run_concept(concept: str, model, tokenizer) -> dict:
    timings = {}

    def timed(name, fn, *args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start)
        return result

    spec = CONCEPTS[concept]
    holdout_prompts = spec["holdout_prompts"]

    fit_pos, fit_neg, n_extractions = _extract_fit_activations(model, tokenizer, spec, timed)

    scan_results = timed("probe_scan", scan_layers, fit_pos, fit_neg)
    auroc_by_layer = {l: round(scan_results[l]["auroc"], 4) for l in sorted(scan_results)}
    print(f"[{concept}] auroc_by_layer (descriptive only, LAYER={LAYER} is fixed): {auroc_by_layer}", flush=True)

    probe, held_out_auroc = timed("final_probe_fit", fit_final_probe, fit_pos, fit_neg, LAYER)
    shuffled_auroc = timed("shuffle_control", shuffle_label_control, fit_pos, fit_neg, LAYER)
    print(f"[{concept}] layer={LAYER} held_out_auroc={held_out_auroc:.3f} "
          f"shuffle_control_auroc={shuffled_auroc:.3f}", flush=True)
    if concept == "non_english" and held_out_auroc < 0.95:
        print(f"[{concept}] WARNING: held_out_auroc at layer {LAYER} is not very high for a concept "
              f"this surface-level -- suspect the extraction, not the concept.", flush=True)

    direction = diff_in_means(fit_pos, fit_neg, LAYER)
    layer_acts = torch.cat([fit_pos[LAYER], fit_neg[LAYER]])
    mean_residual_norm = layer_acts.norm(dim=-1).mean().item()
    print(f"[{concept}] mean_residual_norm at layer {LAYER}: {mean_residual_norm:.1f}", flush=True)

    n_generations = 0
    all_prompts, all_completions, per_alpha_slices = [], [], []
    for multiplier in ALPHA_MULTIPLIERS:
        alpha = multiplier * mean_residual_norm
        start_idx = len(all_completions)
        for i, prompt in enumerate(holdout_prompts):
            completion = timed(
                "steer_generate", generate_steered,
                model, tokenizer, prompt, LAYER, direction, alpha,
            )
            n_generations += 1
            all_prompts.append(prompt)
            all_completions.append(completion)
            print(f"[{concept}] multiplier={multiplier} alpha={alpha:.1f} "
                  f"[{i + 1}/{len(holdout_prompts)}]", flush=True)
        per_alpha_slices.append((multiplier, alpha, slice(start_idx, len(all_completions))))

    print(f"[{concept}] scoring {len(all_completions)} completions with the external judge...", flush=True)
    graded_scores = timed("api_judge_scoring", score_batch, all_completions, concept)

    print(f"[{concept}] measuring unsteered probe response on completions "
          f"(no steering hook active for this pass)...", flush=True)
    probe_responses = [
        timed("probe_response", probe_response, model, tokenizer, probe, LAYER, prompt, completion)
        for prompt, completion in zip(all_prompts, all_completions)
    ]

    gate2_gap = None
    if concept == "refusal":
        keyword_flags = [is_refusal(c) for c in all_completions]
        print(f"[{concept}] Gate 2 (judge vs. keyword anchor):", flush=True)
        gate2_gap = validate_against_keyword(all_completions, keyword_flags, concept)

    sweep_results = []
    for multiplier, alpha, sl in per_alpha_slices:
        scores = [s for s in graded_scores[sl] if s is not None]
        responses = probe_responses[sl]
        sweep_results.append({
            "alpha_multiplier": multiplier,
            "alpha": alpha,
            "mean_graded_score": sum(scores) / len(scores) if scores else None,
            "mean_probe_response": sum(responses) / len(responses),
            "examples": [
                {"prompt": p, "completion": c, "graded_score": s, "probe_response": r}
                for p, c, s, r in zip(
                    all_prompts[sl], all_completions[sl], graded_scores[sl], responses,
                )
            ],
        })
        print(f"[{concept}] multiplier={multiplier} done: "
              f"mean_graded_score={sweep_results[-1]['mean_graded_score']} "
              f"mean_probe_response={sweep_results[-1]['mean_probe_response']:.2f}", flush=True)

    baseline = next(r for r in sweep_results if r["alpha_multiplier"] == 0.0)
    print(f"[{concept}] baseline (alpha=0) mean_graded_score={baseline['mean_graded_score']}", flush=True)
    if concept == "deception" and (baseline["mean_graded_score"] or 0) < 0.15:
        print(f"[{concept}] WARNING: baseline deception score is near floor -- there's no deceptive "
              f"behaviour here to move under steering. The holdout prompts likely need rewriting, "
              f"not the steering.", flush=True)

    time_per_extraction = timings["extract"] / n_extractions
    time_per_generation = timings["steer_generate"] / n_generations
    time_per_judge_call = timings["api_judge_scoring"] / n_generations
    time_per_probe_response = timings["probe_response"] / n_generations

    result = {
        "concept": concept,
        "layer": LAYER,
        "auroc_by_layer": auroc_by_layer,
        "held_out_auroc": held_out_auroc,
        "shuffle_control_auroc": shuffled_auroc,
        "mean_residual_norm": mean_residual_norm,
        "gate2_keyword_agreement_gap": gate2_gap,
        "timings_sec": timings,
        "n_extractions": n_extractions,
        "n_generations": n_generations,
        "time_per_extraction_sec": time_per_extraction,
        "time_per_generation_sec": time_per_generation,
        "time_per_judge_call_sec": time_per_judge_call,
        "time_per_probe_response_sec": time_per_probe_response,
        "sweep": sweep_results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{concept}_run.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[{concept}] wrote {out_path}", flush=True)
    return result


def main():
    gate1_refusal_score, gate1_comply_score = check_judge_separation("refusal")
    print(f"Gate 1 (judge sanity, no GPU yet): hard-refusal={gate1_refusal_score} "
          f"full-compliance={gate1_comply_score}", flush=True)
    gate1_ok = (
        gate1_refusal_score is not None and gate1_comply_score is not None
        and gate1_refusal_score - gate1_comply_score > 0.5
    )
    if not gate1_ok:
        print("Gate 1 FAILED -- judge does not separate hand-written extremes. "
              "Stopping before loading the model.", flush=True)
        return
    print("Gate 1 passed.", flush=True)

    model, tokenizer = load_model()
    print("model loaded", flush=True)

    for concept in CONCEPTS_TO_RUN:
        print(f"=== {concept} ===", flush=True)
        run_concept(concept, model, tokenizer)


if __name__ == "__main__":
    main()
