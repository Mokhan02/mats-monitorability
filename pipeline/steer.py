from contextlib import contextmanager

import torch

from .model import decoder_layer


@contextmanager
def apply_steering(model, layer_idx: int, direction: torch.Tensor, alpha: float):
    """Add alpha * direction to the residual stream at every token position,
    at the decoder block whose output is hidden_states[layer_idx].
    Negative alpha suppresses the direction; positive alpha amplifies it.
    """
    direction = direction.to(model.device, model.dtype)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + alpha * direction,) + output[1:]
        return output + alpha * direction

    handle = decoder_layer(model, layer_idx).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def generate_steered(model, tokenizer, prompt: str, layer_idx: int, direction: torch.Tensor,
                      alpha: float, max_new_tokens: int = 150) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with apply_steering(model, layer_idx, direction, alpha):
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
