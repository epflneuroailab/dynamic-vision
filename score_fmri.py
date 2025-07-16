import src.config

import argparse
import numpy as np
from brainscore_vision.model_helpers.brain_transformation.temporal import iterable_to_list
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, average_repetition, apply_keep_attrs

from src.models.loading import load_model
from src.data import data_store, DATASETS
from src.store import pickle_store, activation_store
from src.evaluate import evaluate_fmri
from src.ceiling import ceiler

score_store = pickle_store.add_node("scores")
ceil_store = pickle_store.add_node("ceiling")
meta_store = pickle_store.add_node("fmri_meta")
layer_store = pickle_store.add_node("layers")
control_store = pickle_store.add_node("control")

def main(args):
    for dataset in args.datasets:
        run_ceiling(args, dataset)
        run_activation(args, dataset)
    run_evaluate(args)


def run_ceiling(args, dataset):
    ceiling_id = f"{dataset}"
    if ceil_store.exists(ceiling_id) and not args.rerun_ceiling:
        return

    assembly = data_store.load(dataset)
    ceiling = ceiler(assembly)
    ceil_store.store(ceiling, ceiling_id)


def run_activation(args, dataset):
    activation_id = f"{args.model}.{dataset}.{args.inference_mode}.{args.context_duration}"
    if activation_store.exists(activation_id) and not args.rerun_activation:
        print(f"Skipping {activation_id}")
        return

    if args.rerun_activation:
        activation_store.clear(activation_id)
        
    ## Load temporal stimuli: a movie clip and neural responses (fMRI:fsaverage5) over time
    assembly = load_average_assembly(dataset, args)

    # Load the model
    # 1. feature downsampled
    # 2. all layers recorded
    # 3. image to temporal conversion
    model, layers = load_model(
                args.model, 
                context_duration=args.context_duration, 
                inference_mode=args.inference_mode,
                batchsize=args.batchsize
            )

    layer_store.store(layers, args.model)

    # Store control parameters
    controls = {}
    controls['fps'] = model._extractor.inferencer.fps
    if hasattr(model, '_model') and hasattr(model._model, 'parameters'):
        controls['params'] = sum(p.numel() for p in model._model.parameters())
    else:
        controls['params'] = 0
    control_store.store(controls, args.model)

    ## Get the model's activations over time
    activations = model(assembly.stimulus_set, layers)
    
    ## Store the activations
    activation_store.store(model._extractor.identifier, assembly.stimulus_set.identifier, activation_id)


def run_evaluate(args):
    datasets_codes = [DATASETS.index(dataset) for dataset in args.datasets]
    datasets_codes.sort()
    datasets_id = "".join([str(code) for code in datasets_codes])
    eval_id = f"fmri.{args.model}.{datasets_id}.{args.inference_mode}.{args.context_duration}.{args.clip_duration}{'.hrf' if args.hrf else ''}"
    if score_store.exists(eval_id) and not args.rerun_evaluation:
        score = score_store.load(eval_id)
        return

    activations = {}
    assemblies = {}
    for dataset in args.datasets:
        activation_id = f"{args.model}.{dataset}.{args.inference_mode}.{args.context_duration}"
        activations[dataset] = activation_store.load(activation_id)
        assemblies[dataset] = load_average_assembly(dataset)
        layers = layer_store.load(args.model)

    test_scores, valid_scores = evaluate_fmri(activations, assemblies, layers, args.clip_duration, hrf=args.hrf)
    score_store.store((test_scores, valid_scores, layers), eval_id)


def load_average_assembly(dataset, args=None):
    data_cache = data_store.add_node("cache")
    if data_cache.exists(dataset) and (args and not args.rerun_all):
        data = data_cache.load(dataset)
    else:
        assembly = data_store.load(dataset)
        assembly = average_repetition(assembly)
        data_cache.store(assembly, dataset)
        data = assembly
    if not meta_store.exists(dataset):
        meta = data.sizes
        meta_store.store(meta, dataset)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a model on a dataset")
    parser.add_argument("--model", "-m", type=str, help="Model name")
    parser.add_argument("--datasets", "-d", type=str, nargs="+", help="Dataset names", default=DATASETS)
    parser.add_argument("--context-duration", type=float, default=4000, help="Duration of the context window")
    parser.add_argument("--clip-duration", type=int, default=15, help="Duration of the clip (sec) for evaluation stage")
    parser.add_argument("--no-hrf", action="store_true", help="Not use HRF")
    parser.add_argument("--batchsize", type=int, default=4, help="Batch size")
    parser.add_argument("--inference-mode", type=str, default="block", help="Inference mode")
    parser.add_argument("--rerun-activation", action="store_true", help="Rerun the model")
    parser.add_argument("--rerun-evaluation", action="store_true", help="Rerun the model")
    parser.add_argument("--rerun-ceiling", action="store_true", help="Rerun the model")
    parser.add_argument("--rerun-all", action="store_true", help="Rerun the model")
    args = parser.parse_args()
    if args.rerun_all:
        args.rerun_activation = True
        args.rerun_evaluation = True
        args.rerun_ceiling = True
    if args.datasets == ["all"]:
        args.datasets = [d for d in DATASETS if d != "fake-fmri"]
    args.hrf = not args.no_hrf
    main(args)
