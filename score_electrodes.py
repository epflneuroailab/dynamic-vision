import src.config

import argparse
import numpy as np
from brainscore_vision.model_helpers.brain_transformation.temporal import iterable_to_list
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, apply_keep_attrs

from src.models.loading import load_model
from src.data import data_store, ELECTRODES
from src.store import pickle_store, activation_store
from src.evaluate import evaluate_electrodes
from src.ceiling import ceiler
from src.utils import average_repetition

score_store = pickle_store.add_node("scores")
ceil_store = pickle_store.add_node("ceiling")
meta_store = pickle_store.add_node("electrodes_meta")
layer_store = pickle_store.add_node("layers")


def main(args):
    print(args)
    for dataset in args.datasets:
        run_ceiling(args, dataset)
        run_activation(args, dataset)
        run_evaluate(args, dataset)


def run_ceiling(args, dataset):
    ceiling_id = f"{dataset}"
    if ceil_store.exists(ceiling_id) and not args.rerun_ceiling:
        return

    if dataset == "crcns-mt2":
        ceiling = None
    else:
        assembly = data_store.load(dataset)
        ceiling = ceiler(assembly)
    ceil_store.store(ceiling, ceiling_id)


def run_activation(args, dataset):
    activation_id = f"{args.model}.{dataset}.{args.inference_mode}"
    if activation_store.exists(activation_id) and not args.rerun_activation:
        print(f"Skipping {activation_id}")
        return

    if args.rerun_activation:
        activation_store.clear(activation_id)
        
    ## Load temporal stimuli: a movie clip and neural responses (fMRI:fsaverage5) over time
    assembly = load_average_assembly(dataset, args)
    time_bin_end = assembly.time_bin_end.max().item()

    # Speed up running by only running the necessary timebins
    if dataset in [
        "freemanziemba2013-V1",
        "freemanziemba2013-V2",
        "majajhong2015-V4",
        "majajhong2015-IT",
    ]:
        context_duration = (time_bin_end, np.inf) 
        temporal_context_strategy = "conservative"
    else:
        context_duration = args.context_duration
        temporal_context_strategy = "greedy"

    ## Load the model
    # 1. feature downsampled
    # 2. all layers recorded
    # 3. image to temporal conversion
    model, layers = load_model(
                args.model, 
                context_duration=context_duration, 
                inference_mode=args.inference_mode,
                batchsize=args.batchsize,
                temporal_context_strategy=temporal_context_strategy
            )

    layer_store.store(layers, args.model)

    ## Get the model's activations over time
    activations = model(assembly.stimulus_set, layers)
    
    ## Store the activations
    activation_store.store(model._extractor.identifier, assembly.stimulus_set.identifier, activation_id)


def run_evaluate(args, dataset):
    layers = layer_store.load(args.model)

    datasets_id = ELECTRODES.index(dataset)
    eval_id = f"elec.{args.model}.{datasets_id}.{args.inference_mode}"

    if score_store.exists(eval_id) and not args.rerun_evaluation:
        score = score_store.load(eval_id)
        print(f"{dataset} : {score[0].mean()}")
        return

    activation_id = f"{args.model}.{dataset}.{args.inference_mode}"
    activation = activation_store.load(activation_id)
    assembly = load_average_assembly(dataset)

    mt_benchmark = dataset in ["crcns-mt2"]
    test_scores, valid_scores = evaluate_electrodes(activation, assembly, layers, mt_benchmark=mt_benchmark)
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
    parser.add_argument("--datasets", "-d", type=str, nargs="+", help="Dataset names", default=ELECTRODES)
    parser.add_argument("--batchsize", type=int, default=4, help="Batch size")
    parser.add_argument("--context-duration", type=float, default=4000, help="Context duration")
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
        args.datasets = [d for d in ELECTRODES if d != "fake-fmri"]
    main(args)
