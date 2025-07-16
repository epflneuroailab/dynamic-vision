import scipy
import numpy as np
from functools import partial

from .utils import *
from . import fmri, task, behaviour, electrodes
from .decoder import decoders
from .behaviour import metrics


def evaluate_fmri(activations, assemblies, layers, clip_duration, decoder_name='ridgecv', delay=5000, n_splits=10, hrf=False):
    assemblies = make_stimulus_paths(assemblies)
    benchmarks = activations.keys()
    delay = 0 if hrf else delay
    activations = {
        benchmark: time_delay(activations[benchmark], delay) 
        if benchmark not in ["mcmahon2023-fmri", "lahner2024-fmri"] else activations[benchmark]
        for benchmark in benchmarks
    }
    for benchmark in benchmarks:
        activations[benchmark], assemblies[benchmark] = time_align(activations[benchmark], assemblies[benchmark])
    if hrf:
        activations = {
            benchmark: time_hrf(activations[benchmark], tr=1) 
            if benchmark not in ["mcmahon2023-fmri", "lahner2024-fmri"] else activations[benchmark]
            for benchmark in benchmarks
        }

    metric = decoders[decoder_name]

    valid_scores, test_scores = fmri.evaluate_per_layer(
        activations, assemblies, layers, n_splits=n_splits,
        clip_duration=clip_duration, metric=metric
    )

    return test_scores, valid_scores


def evaluate_electrodes(activation, assembly, layers, mt_benchmark=False, decoder_name='ridgecv', n_splits=10):
    assembly = make_stimulus_paths({"*": assembly})["*"]
    activation = repeat_last_time_bin(activation)  # to avoid last-time-bin problem
    activation, assembly = time_align(activation, assembly)
    metric = decoders[decoder_name]
    valid_scores, test_scores = electrodes.evaluate_per_layer(
        activation, assembly, layers, n_splits=n_splits, metric=metric, mt_benchmark=mt_benchmark
    )

    return test_scores, valid_scores


def evaluate_task(activation, assembly, layers, decoder_name, n_splits=10):
    assembly = make_stimulus_paths({"*": assembly})["*"]
    valid_scores, test_scores = task.evaluate_per_layer(
        activation, assembly, layers, metric=decoders[decoder_name],
        n_splits=n_splits
    )
    return test_scores, valid_scores


def evaluate_behaviour(assembly, testing_activation, testing_stimulus_set, fitting_activation, fitting_stimulus_set, layers, metric_name, decoder_name="logistic_regression"):
    testing_assembly = make_fitting_assembly(testing_stimulus_set)
    fitting_assembly = make_fitting_assembly(fitting_stimulus_set)
    assembly = make_stimulus_paths({"*": assembly})["*"]

    metric = metrics[metric_name]
    # decoding_scores, = task.evaluate_per_layer(fitting_activation, fitting_assembly, layers, metric=decoders[decoder_name], train_valid_test_ratios=(0.5, 0.5))
    alignment_scores = behaviour.evaluate_per_layer(assembly, testing_activation, testing_assembly, fitting_activation, fitting_assembly, layers, metric=metric)

    return alignment_scores
