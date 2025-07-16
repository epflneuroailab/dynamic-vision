import os
import torch

from ...store import pickle_store


extra_store = pickle_store.add_node("extra")

def map_model_id(identifier):
    tmp = identifier.split("-")
    base_model = tmp[1]
    if base_model == "S3DHT":
        model_name = "s3d-HowTo100M"
    elif base_model == "Uniformer":
        model_name = "UniFormer-V1"
    elif base_model == "VideoMAE":
        model_name = "VideoMAE-V1-L"
    elif base_model == "VJEPA":
        model_name = "VJEPA-Temporal"
    else:
        raise ValueError(f"Unknown model identifier: {identifier}")
    finetune_identifier = identifier
    return finetune_identifier, model_name


def load_finetune_weights(model, finetune_identifier):
    tmp = finetune_identifier.split("-")
    base_model = tmp[1]
    finetune_path = "-".join(tmp[1:])+".pth"
    finetune_path = finetune_path.replace("@", ".").replace("._", "@_")
    weight_path = extra_store._get_path(finetune_path)
    weight = torch.load(weight_path, map_location="cpu")

    if base_model in ["S3DHT"]:
        msg = model.load_state_dict(weight, strict=False)
    elif base_model in ["VJEPA"]:
        weight = {"encoder."+k: v for k, v in weight.items() if k.startswith("encoder.")}
        msg = model.load_state_dict(weight, strict=False)
    elif base_model in ["Uniformer", "VideoMAE"]:
        weight = {k[len("encoder."):]: v for k, v in weight.items() if k.startswith("encoder.")}
        msg = model.load_state_dict(weight, strict=False)
    print(msg)

    return model