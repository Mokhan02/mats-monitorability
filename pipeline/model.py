import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-8B"


def load_model(model_name: str = DEFAULT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def decoder_layer(model, layer_idx: int):
    """hidden_states[l] from output_hidden_states is the residual stream *after*
    decoder block (l-1); hidden_states[0] is the embedding output. So the hook
    for hidden_states layer `l` (l >= 1) belongs on decoder block `l - 1`.
    """
    return model.model.layers[layer_idx - 1]
