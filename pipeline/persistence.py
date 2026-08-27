"""Core persistence measurement: compare clean vs. steered residual streams
at every layer from the injection point onward, for a given steering vector
and coefficient.

probe_delta (probe_j's output on steered minus clean, at every layer j) is
the primary readout: it measures whether the *concept* the intervention was
meant to install survives, not merely whether some perturbation survives.
delta_norm is secondary and separates "the model corrected the concept" from
"the perturbation was damped generically."
"""

import torch

from .model import decoder_layer, encode_chat
from .steer import apply_steering


@torch.no_grad()
def clean_activations(model, tokenizer, prompts: list[str]) -> list[list[torch.Tensor]]:
    """One unsteered forward pass per prompt, capturing every layer at once
    (a single pass, not one pass per layer). Condition-independent -- doesn't
    depend on injection layer, vector, or coefficient -- so compute this once
    per prompt set and reuse across every measurement rather than recomputing
    it inside each call to measure().
    """
    all_acts = []
    for prompt in prompts:
        encoded = encode_chat(tokenizer, prompt, model.device)
        out = model(**encoded, output_hidden_states=True)
        all_acts.append([hs[0, -1, :].float().cpu() for hs in out.hidden_states])
    return all_acts


@torch.no_grad()
def measure(model, tokenizer, prompts: list[str], clean_acts: list[list[torch.Tensor]],
            inject_layer: int, v: torch.Tensor, coeff: float, mean_residual_norm: float,
            probes: dict[int, object]) -> dict[int, dict[str, float]]:
    """Steer at `inject_layer` with `coeff * mean_residual_norm * unit(v)` and,
    for every layer j in inject_layer..n_layers, average over `prompts`:
      delta_norm: ||steered[j] - clean[j]|| / ||clean[j]||
      probe_delta: probes[j](steered[j]) - probes[j](clean[j])  (primary)
      cos_to_v: cosine of (steered[j] - clean[j]) with the injected direction

    `probes` is {layer: fitted classifier}, one per layer, fit once up front
    on the same data with the same regularization -- not re-tuned here.
    `clean_acts` must already be one entry per prompt (see clean_activations).
    """
    n_layers = model.config.num_hidden_layers
    layers = list(range(inject_layer, n_layers + 1))
    sums = {j: {"delta_norm": 0.0, "probe_delta": 0.0, "cos_to_v": 0.0} for j in layers}

    alpha = coeff * mean_residual_norm
    v_unit = v / v.norm()
    hook_module = decoder_layer(model, inject_layer)

    for prompt, clean in zip(prompts, clean_acts):
        assert len(hook_module._forward_hooks) == 0, "hook already attached before steered pass"

        encoded = encode_chat(tokenizer, prompt, model.device)
        with apply_steering(model, inject_layer, v_unit, alpha):
            out = model(**encoded, output_hidden_states=True)
        steered = [hs[0, -1, :].float().cpu() for hs in out.hidden_states]

        assert len(hook_module._forward_hooks) == 0, "hook not removed after steered pass"

        for j in layers:
            c, s = clean[j], steered[j]
            d = s - c
            sums[j]["delta_norm"] += (d.norm() / c.norm()).item()

            probe = probes[j]
            act_c = c.numpy().reshape(1, -1)
            act_s = s.numpy().reshape(1, -1)
            sums[j]["probe_delta"] += float(
                probe.decision_function(act_s)[0] - probe.decision_function(act_c)[0]
            )

            d_norm = d.norm().item()
            sums[j]["cos_to_v"] += (torch.dot(d, v_unit).item() / d_norm) if d_norm > 1e-8 else 0.0

    n = len(prompts)
    return {j: {k: total / n for k, total in sums[j].items()} for j in layers}
