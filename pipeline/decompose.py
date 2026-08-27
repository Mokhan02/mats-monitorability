"""Decompose a probe direction into components parallel and orthogonal to
difference-in-means, to test whether persistence is carried specifically by
the parallel component (the hypothesis: a probe is fit to discriminate, so
its weight vector spends most of its norm on directions that separate the
classes but that downstream computation never reads; diff-in-means captures
the direction the model actually writes).
"""

import torch


def decompose(probe_vec: torch.Tensor, dim_vec: torch.Tensor) -> dict:
    """probe_vec, dim_vec: directions at the same layer (need not be unit
    already; both are unit-normalized here).

    Returns:
      parallel: unit vector -- the component of probe_vec along dim_vec,
        renormalized to unit. Since this is a projection onto a 1-D
        subspace, it is always exactly +dim_vec or -dim_vec.
      orthogonal: unit vector -- probe_vec minus its projection onto
        dim_vec, renormalized to unit.
      cos: cosine similarity between probe_vec and dim_vec.
      orthogonal_norm_fraction: fraction of probe_vec's (unit) norm lying
        orthogonal to dim_vec -- i.e. sqrt(1 - cos^2). Together with cos,
        this is the headline diagnostic.
    """
    probe_vec = probe_vec / probe_vec.norm()
    dim_vec = dim_vec / dim_vec.norm()

    cos = torch.dot(probe_vec, dim_vec).item()
    if abs(cos) < 0.05:
        print(f"WARNING: |cos(probe_vec, dim_vec)| = {abs(cos):.3f} < 0.05 -- "
              f"probe_vec and dim_vec are nearly orthogonal, so 'parallel' is "
              f"numerically meaningless here (it's just +-dim_vec regardless, "
              f"carrying almost none of probe_vec's actual direction).")

    orth_raw = probe_vec - cos * dim_vec
    orthogonal_norm_fraction = orth_raw.norm().item()

    parallel = dim_vec if cos >= 0 else -dim_vec
    orthogonal = (
        orth_raw / orth_raw.norm() if orthogonal_norm_fraction > 1e-8 else orth_raw
    )

    return {
        "parallel": parallel,
        "orthogonal": orthogonal,
        "cos": cos,
        "orthogonal_norm_fraction": orthogonal_norm_fraction,
    }
