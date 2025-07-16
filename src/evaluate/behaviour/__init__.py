import numpy as np
from xarray import DataArray
from functools import partial
from brainio.assemblies import walk_coords, DataAssembly

from ..utils import pearsonr, take_layers, safe_parallel, copy_random_state, align_stimulus_paths, ScaledLogisticRegression
from . import i1i2


metrics = {}
metrics.update(i1i2.metrics)

def evaluate_per_layer(assembly, testing_activation, testing_assembly, fitting_activation, fitting_assembly, layers, metric, **kwargs):
    from joblib import Parallel, delayed
    testing_activations = [take_layers(testing_activation, [layer]) for layer in layers]
    fitting_activations = [take_layers(fitting_activation, [layer]) for layer in layers]
    rets = safe_parallel(evaluate_metric, [(assembly, testing_acti, testing_assembly, fitting_acti, fitting_assembly, metric) for testing_acti, fitting_acti in zip(testing_activations, fitting_activations)], kwargs)
    alignments = []
    model_accs = []
    for alignment, human_acc, model_acc in rets:
        model_accs.append(model_acc)
        alignments.append(alignment)
    alignments = np.array(alignments)  # [num_layers, split]
    model_accs = np.array(model_accs)  # [num_layers, num_unique_stimuli]
    return alignments, human_acc, model_accs

def evaluate_metric(assembly, testing_activation, testing_assembly, fitting_activation, fitting_assembly, metric):
    """
    Evaluate the metric. Fit a decoder (if needed) to the fitting data, and make predictions on the testing data. Then,
    compare the predictions with the assembly using the metric.
    """

    testing_activation, testing_assembly = align_stimulus_paths(testing_activation, testing_assembly)
    fitting_activation, fitting_assembly = align_stimulus_paths(fitting_activation, fitting_assembly)

    fitting_input = fitting_activation.transpose('stimulus_path', 'neuroid').values
    fitting_output = fitting_assembly.transpose('stimulus_path', 'label').values
    testing_input = testing_activation.transpose('stimulus_path', 'neuroid').values

    classifier = ScaledLogisticRegression(C=1.0)
    classifier.fit(fitting_input, fitting_output.ravel())
    prob = classifier.predict_proba(testing_input)

    assembly = _wrap_assembly(assembly)
    prob_assembly = _wrap_probabilistic_prediction(prob, testing_activation.stimulus_path.values, classifier.classes_)
    alignment_score = metric(prob_assembly, assembly)

    # compute the accuracy
    stimulus_ids = assembly.stimulus_id.values
    human_acc = (assembly.truth.values == assembly).values
    model_acc = (testing_assembly.values.flatten() == prob_assembly.idxmax('choice')).values

    return alignment_score, human_acc, model_acc  # [score]

def _wrap_probabilistic_prediction(prob, stimulus_paths, choices):
    model_assembly = DataAssembly(
        prob, 
        dims=("presentation", "choice"),
        coords={
            "stimulus_path": ('presentation', stimulus_paths),
            "stimulus_id": ('presentation', stimulus_paths),
            "choice": choices
        }
    )

    return model_assembly

def _wrap_assembly(assembly):
    stimulus_paths = assembly.stimulus_path.values
    assembly = assembly.assign_coords(stimulus_id=("stimulus_path", stimulus_paths))
    assembly = assembly.rename(stimulus_path="presentation")
    assembly = assembly.assign_coords(stimulus_path=("presentation", stimulus_paths)).drop('presentation')
    cls_ = assembly.__class__
    assembly = cls_(assembly)
    return assembly