from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.temporal.utils import get_specified_layers
from brainscore_vision.model_interface import BrainModel
from . import model


def commit_model(identifier):
    activations_model=model.get_model(identifier)
    layers=get_specified_layers(activations_model)
    return ModelCommitment(identifier=identifier, activations_model=activations_model, layers=layers)

def commit(identifier):
    return lambda: commit_model(identifier)

for i in range(1_000, 10_000, 1_000):
    model_registry[f"S3D-afd101-0.001-32_{i}"] = commit(f"S3D-afd101-0.001-32_{i}")


for i in range(10_000, 100_000 + 1, 10_000):
    model_registry[f"S3D-afd101-0.001-32_{i}"] = commit(f"S3D-afd101-0.001-32_{i}")