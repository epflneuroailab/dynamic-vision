import src.config

import argparse
import numpy as np
from brainscore_vision.model_helpers.brain_transformation.temporal import iterable_to_list
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, average_repetition, apply_keep_attrs

from src.models.loading import load_model
from src.models.groups import IMAGE_MODELS
from src.data import data_store, BEHAVIOURS, STATIC
from src.store import pickle_store, activation_store
from src.evaluate import evaluate_behaviour
from src.ceiling import ceiler
from src.utils import _efficient_mean_time_bin


score_store = pickle_store.add_node("scores")
layer_store = pickle_store.add_node("layers")
fitting_stimuli_store = data_store.add_node("fitting_stimuli")

def get_fitting_stimuli_name(task_name):
    if "han2024" in task_name:
        return 'han2024-RGB-fitting_stimuli' if 'RGB' in task_name else 'han2024-J-6P-fitting_stimuli'
    else:
        return task_name + "-fitting_stimuli"

def main(args):
    for task in args.tasks:
        fitting_name = get_fitting_stimuli_name(task)
        response_stimuli = data_store.load(task).stimulus_set
        fitting_stimuli = fitting_stimuli_store.load(fitting_name)
        run_activation(args, task, response_stimuli)
        run_activation(args, fitting_name, fitting_stimuli)
    run_evaluate(args)

def get_activation_id(args, task):
    single_image_mode = args.model in IMAGE_MODELS and task in STATIC
    if single_image_mode:
        activation_id = f"{args.model}.{task}"
    else:
        activation_id = f"{args.model}.{task}.{args.inference_mode}.{args.context_duration}"
    return activation_id

def run_activation(args, task, stimulus_set):
    activation_id = get_activation_id(args, task)
    if activation_store.exists(activation_id) and not args.rerun_activation:
        return

    if args.rerun_activation:
        activation_store.clear(activation_id)

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
    activations = model(stimulus_set, layers)
    
    ## Store the activations
    activation_store.store(model._extractor.identifier, stimulus_set.identifier, activation_id)


def run_evaluate(args):
    layers = layer_store.load(args.model)
    for task in args.tasks:
        task_id = list(BEHAVIOURS.keys()).index(task)

        eval_id = f"behaviour.{args.model}.{task_id}.{args.inference_mode}.{args.context_duration}"
        if score_store.exists(eval_id) and not args.rerun_evaluation:
            # score = score_store.load(eval_id)
            # print(f"Score for {task} already exists: {score}")
            # breakpoint()
            continue

        activation_id = get_activation_id(args, task)

        fitting_task = get_fitting_stimuli_name(task)
        fitting_activation_id = get_activation_id(args, fitting_task)

        assembly = data_store.load(task)

        testing_activation = activation_store.load(activation_id)
        fitting_activation = activation_store.load(fitting_activation_id)
        testing_activation = _efficient_mean_time_bin(testing_activation)
        fitting_activation = _efficient_mean_time_bin(fitting_activation)

        testing_stimuli = assembly.stimulus_set
        fitting_stimuli = fitting_stimuli_store.load(fitting_task)
        
        alignment_scores = evaluate_behaviour(
            assembly,
            testing_activation, 
            testing_stimuli, 
            fitting_activation, 
            fitting_stimuli, 
            layers, metric_name=BEHAVIOURS[task])

        results = (alignment_scores, layers)
        score_store.store(results, eval_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a model on a task")
    parser.add_argument("--model", "-m", type=str, help="Model name")
    parser.add_argument("--tasks", "-d", type=str, nargs="+", help="Dataset names", default=BEHAVIOURS.keys())
    parser.add_argument("--context-duration", type=float, default=4000, help="Duration of the context window")
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
    main(args)
