import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-8B"


def load_model(model_name: str = DEFAULT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        # transformers >=5 renamed this kwarg from torch_dtype to dtype.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
    model.eval()
    devices = {p.device for p in model.parameters()}
    print(f"cuda available: {torch.cuda.is_available()}; model device(s): {devices}")
    return model, tokenizer


def encode_chat(tokenizer, prompt: str, device) -> dict[str, torch.Tensor]:
    """apply_chat_template's return shape is version-sensitive (raw tensor vs.
    a BatchEncoding dict) depending on transformers version; return_dict=True
    pins it to the dict form so callers can always **kwargs it into the model.
    """
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    )
    return {k: v.to(device) for k, v in encoded.items()}


def decoder_layer(model, layer_idx: int):
    """hidden_states[l] from output_hidden_states is the residual stream *after*
    decoder block (l-1); hidden_states[0] is the embedding output. So the hook
    for hidden_states layer `l` (l >= 1) belongs on decoder block `l - 1`.
    """
    return model.model.layers[layer_idx - 1]
