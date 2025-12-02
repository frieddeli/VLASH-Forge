"""LoRA and QLoRA utilities for VLASH training."""

from vlash.lora.apply import (
    apply_lora,
    infer_unfreeze_modules_from_patterns,
    is_lora_policy,
    is_qlora_policy,
)
from vlash.lora.checkpoint import (
    clone_and_merge_lora_policy,
    load_lora_adapters,
    merge_lora_into_base,
)
from vlash.lora.logging import (
    ModuleLoRAStat,
    count_parameters,
    log_lora_status,
    owner_from_param_name,
    patternize_name,
)
from vlash.lora.qlora import (
    dequantize_model_4bit,
    get_compute_dtype,
    get_parent_module,
    quantize_model_4bit,
    quantize_peft_model_4bit,
)

__all__ = [
    # apply.py
    "apply_lora",
    "infer_unfreeze_modules_from_patterns",
    "is_lora_policy",
    "is_qlora_policy",
    # checkpoint.py
    "clone_and_merge_lora_policy",
    "load_lora_adapters",
    "merge_lora_into_base",
    # logging.py
    "ModuleLoRAStat",
    "count_parameters",
    "log_lora_status",
    "owner_from_param_name",
    "patternize_name",
    # qlora.py
    "dequantize_model_4bit",
    "get_compute_dtype",
    "get_parent_module",
    "quantize_model_4bit",
    "quantize_peft_model_4bit",
]

