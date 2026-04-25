import torch
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained("google/paligemma-3b-pt-224")
for name, param in model.named_parameters():
    if "patch_embedding.weight" in name:
        storage_size = param.data.storage().nbytes()
        tensor_size = param.data.nbytes()
        print(f"{name}: tensor_size={tensor_size}, storage_size={storage_size}")
