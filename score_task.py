import src.config

import argparse
import numpy as np
import _pickle
from brainscore_vision.model_helpers.brain_transformation.temporal import iterable_to_list
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, average_repetition, apply_keep_attrs

from src.models.loading import load_model
from src.models.groups import IMAGE_MODELS
from src.data import data_store, TASKS, STATIC
from src.store import pickle_store, activation_store
from src.evaluate import evaluate_task
from src.ceiling import ceiler
from src.utils import _efficient_mean_time_bin

score_store = pickle_store.add_node("scores")
layer_store = pickle_store.add_node("layers")


def main(args):
    print(args)
    for task in args.tasks:
        run_activation(args, task)
        run_evaluate(args, task)

def get_activation_id(args, task):
    single_image_mode = args.model in IMAGE_MODELS and task in STATIC
    if single_image_mode:
        activation_id = f"{args.model}.{task}"
    else:
        activation_id = f"{args.model}.{task}.{args.inference_mode}.{args.context_duration}"
    return activation_id

def run_activation(args, task):
    activation_id = get_activation_id(args, task)
    if activation_store.exists(activation_id) and not args.rerun_activation:
        return

    if args.rerun_activation:
        activation_store.clear(activation_id)
    ## Load temporal stimuli: a movie clip and neural responses (fMRI:fsaverage5) over time
    assembly = data_store.load(task)

    ## Load the model
    # 1. feature downsampled
    # 2. all layers recorded
    # 3. image to temporal conversion
    single_image_mode = args.model in IMAGE_MODELS and task in STATIC
    model, layers = load_model(
                args.model, 
                context_duration=args.context_duration,
                inference_mode=args.inference_mode,
                batchsize=args.batchsize,
                single_image_mode=single_image_mode
            )

    layer_store.store(layers, args.model)

    ## Get the model's activations over time
    try:
        activations = model(assembly.stimulus_set, layers)
    except (EOFError, _pickle.UnpicklingError) as e:
        import shutil
        import os
        mmap_home = os.environ.get("MMAP_HOME", None)
        mmap_path = os.path.join(mmap_home, assembly.stimulus_set.identifier, model._extractor.identifier)
        shutil.rmtree(mmap_path)
        activations = model(assembly.stimulus_set, layers)
    
    ## Store the activations
    activation_store.store(model._extractor.identifier, assembly.stimulus_set.identifier, activation_id)


def run_evaluate(args, task):
    layers = layer_store.load(args.model)
    task_id = list(TASKS.keys()).index(task)
    eval_id = f"task.{args.model}.{task_id}.{args.inference_mode}.{args.context_duration}"

    if score_store.exists(eval_id) and not args.rerun_evaluation:
        # score = score_store.load(eval_id)
        return

    time_max = None
    if task == "selfmotion":
        time_max = 1000
    elif task == "mcmahon2023":
        time_max = 3000
    elif task == "ding2012":
        time_max = 2000

    activation_id = get_activation_id(args, task)
    activation = activation_store.load(activation_id)
    activation = _efficient_mean_time_bin(activation, time_max=time_max)
    assembly = data_store.load(task)
    test_score, valid_score = evaluate_task(activation, assembly, layers, decoder_name=TASKS[task])
    results = (test_score, valid_score, layers)
    score_store.store(results, eval_id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a model on a task")
    parser.add_argument("--model", "-m", type=str, help="Model name")
    parser.add_argument("--tasks", "-d", type=str, nargs="+", help="Dataset names", default=TASKS.keys())
    parser.add_argument("--context-duration", type=float, default=4000, help="Duration of the context window")
    parser.add_argument("--batchsize", type=int, default=6, help="Batch size")
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
    main(args)
