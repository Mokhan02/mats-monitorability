"""Layer-span steering persistence.

Goodfire (arXiv 2605.12412, Appendix F) observed that probe-weight steering
at a single layer produces an effect that disappears by later layers, while
difference-in-means steering persists, and called the multi-layer probe
workaround ad hoc without explaining the asymmetry.

Hypothesis: a probe is fit to *discriminate*, so its weight vector spends
most of its norm on directions that separate the classes but that
downstream computation never reads (cf. arXiv 2602.06801's large Jacobian
null space). Diff-in-means captures the direction the model actually writes.

The experiment is a regularization sweep, not a single probe_vec vs.
dim_vec comparison: strong regularization (small C) pulls a probe's weights
toward diff-in-means, weak regularization (large C) lets it pick up
idiosyncratic discriminative directions, so sweeping C gives probes spanning
a range of cos(probe, dim) instead of one fixed point. The prediction is that
persistence declines as collinearity (cos) declines -- a dose-response test,
and one that reconciles Goodfire's result rather than just contradicting it:
if their probes were weakly regularized, they'd sit at low collinearity
where decay is fast.

Extraction is prompt-only (harmful request vs. benign request, no
completions at all): this experiment reads a direction out of the residual
stream directly, there's no monitor and no adversary, so there's nothing for
a completion-based leak to hide in.

No behavioural judging, generation, or scoring of any kind -- the readout is
probe response across layers on clean vs. steered forward passes. The
injection layer is the outer loop here, not swept inside the persistence
measurement itself. The *readout* probes (one per layer, used to compute
probe_delta at every layer from the injection point on) are fit once at a
fixed default C, same regularization throughout, not re-tuned per layer and
not selected by argmax. Only the *steering-vector* probes -- fit once per
injection layer, at several C values -- vary in regularization.
"""

import json
import time
from pathlib import Path

import torch

from .decompose import decompose
from .direction import diff_in_means
from .extract import extract_layer_activations
from .model import load_model
from .pairs import REFUSAL_FIT_PAIRS, REFUSAL_HOLDOUT_PROMPTS
from .persistence import clean_activations, measure
from .probe import fit_final_probe, fit_shuffled_label_probe, scan_layers

MODEL = "Qwen/Qwen3-8B"
INJECT_LAYERS = [12, 18, 24]
# Multiples of the layer's mean residual norm. +-1.0 produced degenerate
# output in earlier work on this project; don't use it.
COEFFS = [0.2, 0.35, 0.5]
N_PROMPTS = 24  # both sides of the 12 held-out refusal pairs

# Small C = strong regularization -> expect high cos(probe, dim). Large C =
# weak regularization -> expect lower cos, more idiosyncratic direction.
PROBE_C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]

SHUFFLE_LABEL_SEED = 0
RANDOM_DIRECTION_SEED_BASE = 0  # actual seed used is this + inject_layer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _extract_fit_activations(model, tokenizer):
    """Prompt-only: harmful request vs. benign request, no completions."""
    pos = extract_layer_activations(model, tokenizer, [p.positive for p in REFUSAL_FIT_PAIRS])
    neg = extract_layer_activations(model, tokenizer, [p.negative for p in REFUSAL_FIT_PAIRS])
    return pos, neg


def _random_direction(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.dot(a / a.norm(), b / b.norm()).item()


def main():
    assert len(REFUSAL_HOLDOUT_PROMPTS) == N_PROMPTS, (
        f"expected {N_PROMPTS} held-out prompts, got {len(REFUSAL_HOLDOUT_PROMPTS)}"
    )

    model, tokenizer = load_model()
    print("model loaded", flush=True)

    fit_pos, fit_neg = _extract_fit_activations(model, tokenizer)

    # Readout probes: one per layer, fixed default C, same regularization
    # throughout -- this is the measurement instrument, not part of the
    # collinearity sweep, and stays fixed regardless of which steering
    # vector/C is under test below.
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

        dim_vec = diff_in_means(fit_pos, fit_neg, L)
        mean_residual_norm = torch.cat([fit_pos[L], fit_neg[L]]).norm(dim=-1).mean().item()
        print(f"[L={L}] mean_residual_norm={mean_residual_norm:.1f}", flush=True)

        # Steering-vector probes: fit ONLY at the injection layer, at each C,
        # to get a range of directions spanning high to low collinearity
        # with dim_vec.
        c_vecs, c_cos = {}, {}
        for c in PROBE_C_VALUES:
            clf, _ = fit_final_probe(fit_pos, fit_neg, L, C=c)
            vec = torch.tensor(clf.coef_[0], dtype=torch.float32)
            c_vecs[c] = vec
            c_cos[c] = _cos(vec, dim_vec)
        print(f"[L={L}] cos_by_C: {c_cos}", flush=True)

        least_collinear_C = min(PROBE_C_VALUES, key=lambda c: abs(c_cos[c]))
        decomp = decompose(c_vecs[least_collinear_C], dim_vec)
        print(f"[L={L}] orthogonal derived from C={least_collinear_C} "
              f"(cos={decomp['cos']:.3f}, orthogonal_norm_fraction={decomp['orthogonal_norm_fraction']:.3f})",
              flush=True)

        shuffled_clf, shuffled_auroc = fit_shuffled_label_probe(
            fit_pos, fit_neg, L, seed=SHUFFLE_LABEL_SEED,
        )
        shuffled_vec = torch.tensor(shuffled_clf.coef_[0], dtype=torch.float32)
        print(f"[L={L}] shuffle_label_control_auroc={shuffled_auroc:.3f} "
              f"(sanity: should sit near 0.5)", flush=True)

        random_vec = _random_direction(dim_vec.shape[0], seed=RANDOM_DIRECTION_SEED_BASE + L)

        conditions = {f"probe_C{c}": c_vecs[c] for c in PROBE_C_VALUES}
        conditions["diff_in_means"] = dim_vec
        conditions["orthogonal"] = decomp["orthogonal"]
        conditions["random"] = random_vec
        conditions["shuffle_label"] = shuffled_vec

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
            "probe_c_values": PROBE_C_VALUES,
            "cos_by_C": {str(c): cos for c, cos in c_cos.items()},
            "least_collinear_C": least_collinear_C,
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
