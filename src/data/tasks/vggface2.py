from pathlib import Path

import pandas as pd
import numpy as np

from brainio.stimuli import StimulusSet, StimulusSetLoader
from brainio.assemblies import DataAssembly
from brainscore_core import Score
from brainscore_vision import load_metric
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.model_interface import BrainModel
import os

SUBSET_CLASS_NUM = 100
MAX_IMAGE_PER_CLASS = 50

vggface2_test_dir = "/home/ytang/workspace/data/vggface2/test"
vggface2_classes = {}
for root, dirs, files in os.walk(vggface2_test_dir):
    for file in files:
        if file.endswith(".jpg"):
            class_name = os.path.basename(root)
            filepath = os.path.join(root, file)
            vggface2_classes.setdefault(class_name, []).append(filepath)

def get_data():
    labels = list(vggface2_classes.keys())
    np.random.seed(42)
    subset_labels = np.random.choice(labels, SUBSET_CLASS_NUM, replace=False)
    
    labels = []
    stimulus_paths = []
    stimulus_ids = []

    for label in subset_labels:
        image_paths = vggface2_classes[label]
        image_paths = np.random.choice(image_paths, MAX_IMAGE_PER_CLASS, replace=False)
        for i, image_path in enumerate(image_paths):
            stimulus_paths.append(image_path)
            stimulus_ids.append(f"{label}_{os.path.basename(image_path)}")
            labels.append(label)

    return stimulus_paths, stimulus_ids, labels


def load_dataset(identifier="vggface2-100"):
    stimulus_paths, stimulus_ids, labels = get_data()
    labels = np.array(labels)

    stimulus_set = {}
    stimulus_set["stimulus_id"] = stimulus_ids
    stimulus_set["action"] = labels
    stimulus_set = StimulusSet(stimulus_set)
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}
    stimulus_set.identifier = identifier

    assembly = DataAssembly(
        labels[:, None],
        dims=('presentation', 'label'),
        coords={
            'stimulus_id': ('presentation', stimulus_ids),
            'action': ('presentation', labels),
        }
    )

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly 