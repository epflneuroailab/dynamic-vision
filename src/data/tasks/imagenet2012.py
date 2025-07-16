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

imagenet_val_dir = "/work/upschrimpf1/akgokce/datasets/imagenet/val"
imagenet_file_map = {}
for root, dirs, files in os.walk(imagenet_val_dir):
    for file in files:
        if file.endswith(".JPEG"):
            filepath = os.path.join(root, file)
            imagenet_file_map[file] = filepath

def get_file_path(filename):
    return imagenet_file_map[filename]


def get_data():
    df = pd.read_csv(Path(__file__).parent / 'imagenet2012.csv')

    labels = df.label.unique().tolist()
    np.random.seed(42)
    subset_labels = np.random.choice(labels, SUBSET_CLASS_NUM, replace=False)
    df = df[df.label.isin(subset_labels)]

    stimulus_paths = [get_file_path(row.filename) for row in df.itertuples()]
    stimulus_ids = df.filename.values
    labels = df.label.values
    return stimulus_paths, stimulus_ids, labels


def load_dataset(identifier="imagenet-100"):
    stimulus_paths, stimulus_ids, labels = get_data()

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