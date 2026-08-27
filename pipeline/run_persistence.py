"""Layer-span steering persistence.

Goodfire (arXiv 2605.12412, Appendix F) observed that probe-weight steering
at a single layer produces an effect that disappears by later layers, while
difference-in-means steering persists, and called the multi-layer probe
workaround ad hoc without explaining the asymmetry.

Hypothesis: a probe is fit to *discriminate*, so its weight vector spends
most of its norm on directions that separate the classes but that
downstream computation never reads (cf. arXiv 2602.06801's large Jacobian
null space). Diff-in-means captures the direction the model actually writes.
Prediction: persistence is carried entirely by the component of the probe
vector parallel to diff-in-means.

No behavioural judging, generation, or scoring of any kind -- the readout is
probe response across layers on clean vs. steered forward passes. The
injection layer is the outer loop here, not swept inside the persistence
measurement itself. Probes are fit at every layer by design (not re-tuned
per injection layer, and not selected by argmax -- there is no selection
step on this branch at all).
"""

import json
import random
import time
from pathlib import Path

import torch

from .decompose import decompose
from .direction import diff_in_means
from .extract import extract_layer_activations_for_answers
from .model import load_model
from .pairs import (
    COMPLIANCE_COMPLETIONS,
    REFUSAL_COMPLETIONS,
    REFUSAL_FIT_PAIRS,
    REFUSAL_HOLDOUT_PROMPTS,
)
from .persistence import clean_activations, measure
from .probe import fit_shuffled_label_probe, scan_layers

MODEL = "Qwen/Qwen3-8B"
INJECT_LAYERS = [12, 18, 24]
# Multiples of the layer's mean residual norm. +-1.0 produced degenerate
# output in earlier work on this project; don't use it.
COEFFS = [0.2, 0.35, 0.5]
N_PROMPTS = 24  # both sides of the 12 held-out refusal pairs

CANNED_COMPLETION_SEED = 0
SHUFFLE_LABEL_SEED = 0
RANDOM_DIRECTION_SEED_BASE = 0  # actual seed used is this + inject_layer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _extract_fit_activations(model, tokenizer):
    """Same canned-completion randomization as main's run_pipeline.py: fixed
    seed, one of several phrasings per pair, not a single repeated string
    (a single fixed completion per class is a lexical leak the probe can
    memorize instead of learning the concept).
    """
    rng = random.Random(CANNED_COMPLETION_SEED)
    pos_examples = [(p.positive, rng.choice(REFUSAL_COMPLETIONS)) for p in REFUSAL_FIT_PAIRS]
    neg_examples = [(p.negative, rng.choice(COMPLIANCE_COMPLETIONS)) for p in REFUSAL_FIT_PAIRS]
    pos = extract_layer_activations_for_answers(model, tokenizer, pos_examples)
    neg = extract_layer_activations_for_answers(model, tokenizer, neg_examples)
    return pos, neg


def _random_direction(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def main():
    assert len(REFUSAL_HOLDOUT_PROMPTS) == N_PROMPTS, (
        f"expected {N_PROMPTS} held-out prompts, got {len(REFUSAL_HOLDOUT_PROMPTS)}"
    )

    model, tokenizer = load_model()
    print("model loaded", flush=True)

    fit_pos, fit_neg = _extract_fit_activations(model, tokenizer)

    # Probes fit at every layer, same data, same regularization throughout --
    # no per-layer tuning, no argmax selection. scan_layers already does
    # exactly this fit-at-every-layer pass; nothing here uses its AUROC to
    # pick a layer.
    scan_results = scan_layers(fit_pos, fit_neg)
    auroc_by_layer = {l: round(scan_results[l]["auroc"], 4) for l in sorted(scan_results)}
    probes = {l: scan_results[l]["probe"] for l in scan_results}
    print(f"auroc_by_layer: {auroc_by_layer}", flush=True)

    clean_acts = clean_activations(model, tokenizer, REFUSAL_HOLDOUT_PROMPTS)
    print(f"clean activations captured for {len(REFUSAL_HOLDOUT_PROMPTS)} held-out prompts "
          f"(computed once, reused across every injection layer/condition/coeff)", flush=True)

    n_layers = model.config.num_hidden_layers

    for L in INJECT_LAYERS:
        t0 = time.perf_counter()

        probe_clf = probes[L]
        probe_vec = torch.tensor(probe_clf.coef_[0], dtype=torch.float32)
        dim_vec = diff_in_means(fit_pos, fit_neg, L)

        decomp = decompose(probe_vec, dim_vec)
        print(f"[L={L}] cos(probe, dim)={decomp['cos']:.3f}  "
              f"orthogonal_norm_fraction={decomp['orthogonal_norm_fraction']:.3f}", flush=True)

        mean_residual_norm = torch.cat([fit_pos[L], fit_neg[L]]).norm(dim=-1).mean().item()
        print(f"[L={L}] mean_residual_norm={mean_residual_norm:.1f}", flush=True)

        shuffled_clf, shuffled_auroc = fit_shuffled_label_probe(
            fit_pos, fit_neg, L, seed=SHUFFLE_LABEL_SEED,
        )
        shuffled_vec = torch.tensor(shuffled_clf.coef_[0], dtype=torch.float32)
        print(f"[L={L}] shuffle_label_control_auroc={shuffled_auroc:.3f} "
              f"(sanity: should sit near 0.5)", flush=True)

        random_vec = _random_direction(probe_vec.shape[0], seed=RANDOM_DIRECTION_SEED_BASE + L)

        conditions = {
            "probe_full": probe_vec,
            "parallel": decomp["parallel"],
            "orthogonal": decomp["orthogonal"],
            "diff_in_means": dim_vec,
            "random": random_vec,
            "shuffle_label": shuffled_vec,
        }

        conditions_result = {name: {} for name in conditions}
        for cond_name, v in conditions.items():
            for coeff in COEFFS:
                print(f"[L={L}] measuring condition={cond_name} coeff={coeff}...", flush=True)
                conditions_result[cond_name][str(coeff)] = measure(
                    model, tokenizer, REFUSAL_HOLDOUT_PROMPTS, clean_acts,
                    L, v, coeff, mean_residual_norm, probes,
                )

        elapsed = time.perf_counter() - t0
        print(f"[L={L}] done in {elapsed:.1f}s", flush=True)

        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = RESULTS_DIR / f"persistence_L{L}.json"
        out_path.write_text(json.dumps({
            "model": MODEL,
            "inject_layer": L,
            "coeffs": COEFFS,
            "n_prompts": N_PROMPTS,
            "auroc_by_layer": auroc_by_layer,
            "cos_probe_dim": decomp["cos"],
            "orthogonal_norm_fraction": decomp["orthogonal_norm_fraction"],
            "shuffle_label_control_auroc": shuffled_auroc,
            "mean_residual_norm": mean_residual_norm,
            "conditions": conditions_result,
        }, indent=2))
        print(f"[L={L}] wrote {out_path}", flush=True)

        # Four-column summary: condition x {L, L+4, L+8, final layer},
        # probe_delta only, at the middle coefficient.
        mid_coeff = COEFFS[len(COEFFS) // 2]
        checkpoints = sorted(set(j for j in (L, L + 4, L + 8, n_layers) if L <= j <= n_layers))
        print(f"[L={L}] probe_delta summary at coeff={mid_coeff}, layers {checkpoints}:", flush=True)
        for cond_name in conditions:
            per_layer = conditions_result[cond_name][str(mid_coeff)]
            row = "  ".join(f"L{j}={per_layer[j]['probe_delta']:+.3f}" for j in checkpoints)
            print(f"  {cond_name:16s} {row}", flush=True)


if __name__ == "__main__":
    main()
