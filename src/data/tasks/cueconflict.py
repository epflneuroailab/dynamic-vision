import os
import numpy as np
import h5py
from brainio.stimuli import StimulusSet
from brainio.assemblies import DataAssembly
from brainscore_vision.model_helpers.activations.temporal.inputs.video import get_video_stats


DATA_DIR = '/home/ytang/workspace/data/shape_bias'


def load_dataset(identifier="cue-conflict"):
    stimulus_paths, stimulus_ids, labels = get_data()

    contents = []
    styles = []
    for stimulus_id in stimulus_ids:
        content, style = stimulus_id.split('.')[0].split('-')
        contents.append(content)
        styles.append(style)

    stimulus_set = {}
    stimulus_set["stimulus_id"] = stimulus_ids
    stimulus_set["object_name"] = labels
    stimulus_set["content"] = contents
    stimulus_set["style"] = styles
    stimulus_set = StimulusSet(stimulus_set)
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}
    stimulus_set.identifier = identifier

    assembly = DataAssembly(
        labels[:, None],
        dims=('presentation', 'label'),
        coords={
            'stimulus_id': ('presentation', stimulus_ids),
            'object_name': ('presentation', labels),
            'content': ('presentation', contents),
            'style': ('presentation', styles),
        }
    )

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly 


def get_data():
    np.random.seed(42)
    image_dir = os.path.join(DATA_DIR, 'cue-conflict')
    stimulus_paths = []
    stimulus_ids = []
    labels = []
    condition_count = 0
    action_dirs = os.listdir(image_dir)
    action_dirs = np.random.permutation(action_dirs)

    for i, action_type in enumerate(action_dirs):
        print(f"{i+1}/{len(action_dirs)} {action_type}", end='\r')
        action_dir = os.path.join(image_dir, action_type)

        stimulus_paths_to_add = []
        stimulus_ids_to_add = []
        labels_to_add = []

        image_files = os.listdir(action_dir)
        image_files = np.random.permutation(image_files)

        count = 0 
        for j, image_name in enumerate(image_files):
            image_path = os.path.join(action_dir, image_name)
            stimulus_paths_to_add.append(image_path)
            stimulus_ids_to_add.append(image_name)
            labels_to_add.append(action_type)
            count += 1

        stimulus_paths.extend(stimulus_paths_to_add)
        stimulus_ids.extend(stimulus_ids_to_add)
        labels.extend(labels_to_add)
        condition_count += 1

    labels = np.array(labels)
    return stimulus_paths, stimulus_ids, labels
