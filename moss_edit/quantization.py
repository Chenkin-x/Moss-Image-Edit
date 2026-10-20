"""Quanto INT8 frozen weights; LoRA and activations remain BF16."""
import torch
from optimum.quanto import quantize,freeze,qint8


def quantize_frozen_linears(model, engine=None):
    trainable=[p for p in model.parameters() if p.requires_grad]
    names=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and not m.weight.requires_grad]
    quantize(model,weights=qint8,include=names)
    freeze(model)
    # Quanto creates fresh Parameters when freezing. Restore exactly the
    # original trainable set so the quantized base never receives gradients.
    model.requires_grad_(False)
    for p in trainable:p.requires_grad_(True)
    if engine is not None:
        # ZeRO-2 synchronized the BF16 model before quantization. Its optimizer
        # owns only LoRA; replace debug/name references to release old weights.
        engine.param_names.clear()
        engine.param_names.update({p:n for n,p in engine.module.named_parameters()})
        from deepspeed.utils.debug import debug_extract_module_and_param_names
        debug_extract_module_and_param_names(engine.module)
    return len(names)


def keep_adapter_state(module, state, prefix, metadata):
    # Quanto serializes a frozen weight as weight._data / weight._scale.
    # DeepSpeed's ordinary exclude_frozen_parameters does not remove those.
    for key in list(state):
        if not key.endswith(('.lora_A','.lora_B')):del state[key]
