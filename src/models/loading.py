import sys
import numpy as np
from io import StringIO
from contextlib import contextmanager
from brainscore_vision import load_model as _brainscore_load_model
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.activations.temporal.core.inferencer import CausalInferencer, BlockInferencer


from .convert_image_model import _image_to_temporal_model
from .sparse_random_projection import _register_downsampling_hook
from .groups import STATIC_MODELS, RECURRENT_MODELS, KADIR_MODELS
from . import extra
from .extra import finetune as finetune_module


def brainscore_load_model(model_identifier):
    try:
        return _brainscore_load_model(model_identifier)
    except:
        return model_registry[model_identifier]()

def load_model(
        model_identifier, context_duration=4000, inference_mode="block", batchsize=4, 
        single_image_mode=False, temporal_context_strategy="greedy", downsample_features=True 
    ):
    print("======== Loading model ========")
    print(f"model_identifier: {model_identifier}")

    random = model_identifier.startswith("Random-")
    if random:
        model_identifier = "-".join(model_identifier.split("-")[1:])

    finetune = model_identifier.startswith("Finetune-")
    if finetune:
        finetune_identifier, model_identifier = finetune_module.map_model_id(model_identifier)

    no_downsample = model_identifier.startswith("NoDownsample-")
    if no_downsample:
        model_identifier = "-".join(model_identifier.split("-")[1:])
        downsample_features = False

    model = brainscore_load_model(model_identifier)
    layers = _get_layers(model) if model_identifier in KADIR_MODELS else model.layers
    if model_identifier == "CORnet-S":
        layers = ["V1", "V2", "V4", "IT", "decoder.avgpool"]

    # convert image models to temporal
    if model_identifier in STATIC_MODELS:
        model = _image_to_temporal_model(model, layers, batchsize)

    # resample the layers so that the total number of layers is not too large
    layers = _resample_layers(layers)
    model.layers = layers

    # retarget the recorded region to all layers
    model.layer_model.region_layer_map = _fake_map(layers)
    model.layer_model._layer_model.region_layer_map = _fake_map(layers)

    # rebind inferencer
    model = _rebind_inferencer(model, inference_mode, context_duration, batchsize, single_image_mode=single_image_mode, temporal_context_strategy=temporal_context_strategy)

    # register hooks for model activation downsampling
    if downsample_features:
        model = _register_downsampling_hook(model)

    print(f"Using {inference_mode} inferencer")
    print("======== Model loaded ========")

    layers = model.layers
    model = model.activations_model

    if random:
        model.identifier = f"Random-{model_identifier}"
        model._extractor._identifier = f"Random-{model_identifier}"
        reinitialize_model(model._model)

    if finetune:
        model.identifier = finetune_identifier
        model._extractor._identifier = finetune_identifier
        finetune_module.load_finetune_weights(model._model, finetune_identifier)

    if no_downsample:
        model.identifier = f"NoDownsample-{model_identifier}"
        model._extractor._identifier = f"NoDownsample-{model_identifier}"

    return model, layers

def _resample_layers(layers, limit=15):
    L = len(layers)
    if L > limit:
        steps = np.linspace(0, L-1e-6, limit)
        indices = steps.astype(int)
        print(f"Resampling layers from {L} to {indices}")
        layers = [layers[i] for i in indices]

    print(f"#layers resampled from {L} to {len(layers)}")
    return layers

class _fake_map:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data

def _get_layers(model):
    python_model = model.layer_model.activations_model._model
    layers = [name for name, _ in python_model.named_modules() if len(name) > 0]
    # skip the first layer, usually not useful for neural prediction
    if len(layers) > 1: layers = layers[1:]  
    # skip the last layer, usually task readout
    if len(layers) > 1: layers = layers[:-1]
    return layers

def _rebind_inferencer(model, inference_mode, context_duration, batchsize, single_image_mode=False, temporal_context_strategy="greedy"):
    activations_model = model.activations_model
    if model.identifier in STATIC_MODELS:
        inferencer_cls = BlockInferencer
    else:
        inferencer_cls = CausalInferencer if inference_mode == "causal" else BlockInferencer
    duration = activations_model._extractor.inferencer.duration
    if isinstance(duration, float):
        duration = context_duration
    else:
        if isinstance(context_duration, (tuple, list)):
            duration = (max(duration[0], context_duration[0]), context_duration[1])
        else:
            duration = (duration[0], context_duration)
        
        duration = list(duration)
        duration[0] = max(duration[0], 1000/activations_model._extractor.inferencer.fps+1e-6)

    kwargs = dict(
        inferencer_cls=inferencer_cls,  # or BlockInferencer here
        layer_activation_format=activations_model._extractor.inferencer.layer_activation_format,  # use the layer spec from the original model
        fps=activations_model._extractor.inferencer.fps,  # here I use the original fps of this model by accessing activations_model._extractor.inferencer.fps
                                                        # one can use any other fps value
        num_frames=activations_model._extractor.inferencer.num_frames,  # use the original num_frames of this model
        duration=duration,
        batch_size=batchsize,
        max_workers=batchsize,
        batch_padding=activations_model._extractor.inferencer._executor.batch_padding,
        temporal_context_strategy=temporal_context_strategy,
        img_duration=duration[0],
    )

    if single_image_mode:
        frame_duration = 1000 / activations_model._extractor.inferencer.fps
        kwargs["duration"] = frame_duration
        kwargs["num_frames"] = 1
        kwargs["img_duration"] = frame_duration

    # constrain huge models
    if model.identifier == "MIM": kwargs["num_frames"] = 10

    activations_model.build_extractor(**kwargs)   
    return model

def reinitialize_model(model):

    import torch
    import torch.nn as nn
    import math

    def kaiming_init(module):
        """Applies Kaiming initialization to all applicable layers in a PyTorch module."""
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            # For Conv2d and Conv3d layers
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            # For Linear (fully connected) layers
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)

        elif isinstance(module, (nn.LayerNorm, nn.Embedding, nn.BatchNorm2d, nn.BatchNorm3d)):
            # reset LayerNorm and Embedding layers
            module.reset_parameters()

    model.apply(kaiming_init)