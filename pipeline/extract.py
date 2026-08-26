import torch

from .model import encode_chat


@torch.no_grad()
def extract_layer_activations(model, tokenizer, prompts: list[str]) -> dict[int, torch.Tensor]:
    """One forward pass per prompt, last-token residual stream at every layer.

    Returns dict[layer_idx] -> [n_prompts, hidden_dim] float32 tensor on CPU.
    layer_idx runs 0..num_hidden_layers, matching output_hidden_states indexing
    (0 = embeddings, l = output of decoder block l-1).
    """
    n_layers = model.config.num_hidden_layers
    acts: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers + 1)}

    for prompt in prompts:
        encoded = encode_chat(tokenizer, prompt, model.device)
        out = model(**encoded, output_hidden_states=True)
        for l, hs in enumerate(out.hidden_states):
            acts[l].append(hs[0, -1, :].float().cpu())

    return {l: torch.stack(v) for l, v in acts.items()}
