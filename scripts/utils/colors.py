from src.models.groups import *


BASELINE_1 = "#77767B"
BASELINE_2 = "#3D3846"
MODELS = [
    "#613583",  # purple
    "#99C1F1",  # light blue
    "#61A1EA",  # blue
    "#3584E4",  # dark blue
    "#1D72D8",  # darker blue
    "#1B5FB4",  # even darker blue
]

STATIC = "#3385BC" # blue
DYNAMIC = "#C42C4C" # red
MIXED = "#FDCF7D"

HUMAN = "#29A069"
MODEL = "#1D65C3"


def get_color_by_model(model):
    if model in ["pixels", "hmax", "motion-energy", "MotionNet"]:
        return BASELINE_1
    elif model in RANDOM_MODELS:
        return BASELINE_2
    elif model in IMAGE_MODELS:
        return MODELS[0]
    elif model in FORWARD_PREDICTION_MODELS:
        return MODELS[1]
    elif model in AUDIO_VIDEO_MODELS:
        return MODELS[2]
    elif model in ACTION_RECOGNITION_MODELS:
        return MODELS[3]
    elif model in MASKED_AUTOENCODER_MODELS:
        return MODELS[4]
    elif model in TEXT_VIDEO_MODELS:
        return MODELS[5]
    else:
        return "#000000"
    
MODEL_OF_INTEREST = [
    'VideoMAE-V1-L',
    'VJEPA-Temporal',

    'UniFormer-V1',
    'VideoSwin-L',
    
    'I3D-nonlocal',
]