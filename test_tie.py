import torch
from transformers import AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained("google/paligemma-3b-pt-224")

for k, v in model.named_parameters():
    if "patch_embedding" in k or "embed_tokens" in k:
        print(k, "data_ptr:", v.data_ptr(), "storage_data_ptr:", v.storage().data_ptr(), "tensor_size:", v.nbytes, "storage_size:", v.storage().nbytes())
