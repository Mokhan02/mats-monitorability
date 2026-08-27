import torch

from .model import encode_chat, encode_full_turn


def _extract(model, tokenizer, items, encode_fn) -> dict[int, torch.Tensor]:
    """One forward pass per item, last-token residual stream at every layer.

    Returns dict[layer_idx] -> [n_items, hidden_dim] float32 tensor on CPU.
    layer_idx runs 0..num_hidden_layers, matching output_hidden_states indexing
    (0 = embeddings, l = output of decoder block l-1).
    """
    n_layers = model.config.num_hidden_layers
    acts: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers + 1)}

    for item in items:
        encoded = encode_fn(tokenizer, item, model.device)
        out = model(**encoded, output_hidden_states=True)
        for l, hs in enumerate(out.hidden_states):
            acts[l].append(hs[0, -1, :].float().cpu())

    return {l: torch.stack(v) for l, v in acts.items()}


@torch.no_grad()
def extract_layer_activations(model, tokenizer, prompts: list[str]) -> dict[int, torch.Tensor]:
    """Prompt-contrast extraction (refusal): activations right before
    generation would begin, no completion present. Used where the contrast
    lives in the request itself.
    """
    return _extract(model, tokenizer, prompts, lambda tok, prompt, dev: encode_chat(tok, prompt, dev))


@torch.no_grad()
def extract_layer_activations_for_answers(
    model, tokenizer, examples: list[tuple[str, str]],
) -> dict[int, torch.Tensor]:
    """Answer-contrast extraction (deception, non_english): activations at
    the end of a full (question, answer) turn. Used where the contrast lives
    in how the question gets answered, not in the question itself.
    """
    return _extract(
        model, tokenizer, examples,
        lambda tok, item, dev: encode_full_turn(tok, item[0], item[1], dev),
    )
