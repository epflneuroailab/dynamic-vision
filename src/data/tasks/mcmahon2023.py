import os
from brainio.assemblies import DataAssembly
from brainio.stimuli import StimulusSet
import numpy as np
from joblib import Parallel, delayed
from brainscore_vision.model_helpers.activations.temporal.utils import stack_with_nan_padding
from brainscore_vision.metrics import Score
from brainscore_vision.metric_helpers.transformations import apply_aggregate, standard_error_of_the_mean


SOCIAL_INTERACTION = ['joint action', 'communication']


def load_stimulus_set(identifier='McMahon2023-fMRI'):
    stimulus_ids, stimulus_paths = get_video_paths()
    annos = get_annotations(stimulus_ids)
    stimulus_set = StimulusSet([{'stimulus_path': path, "stimulus_id": i, **annos[i]} 
                                for i, path in zip(stimulus_ids, stimulus_paths)])
    stimulus_set.identifier = f'McMahon2023-fMRI'
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}
    return stimulus_set, stimulus_ids, annos


def load_dataset(identifier='McMahon2023-behaviour'):
    stimulus_set, stimulus_ids, annos = load_stimulus_set(identifier)

    data = np.array([[annos[sid][k] for k in SOCIAL_INTERACTION] for sid in stimulus_ids])

    assembly = DataAssembly(
        data,
        dims=('stimulus_id', 'label'),
        coords={
            'stimulus_id': ('stimulus_id', stimulus_ids),
            'label': ('label', SOCIAL_INTERACTION),
        }
    )

    assembly = assembly.stack(presentation=("stimulus_id", ))

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly


def _get_files(csv):
    with open(csv, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines[1:]]
    return lines


# Annotations:
#   'indoor'
#   'expanse'
#   'transitivity'
#   'agent distance'
#   'facingness'
#   'joint action'
#   'communication'
#   'cooperation'
#   'dominance'
#   'intimacy'
#   'valence'
#   'arousal'
def get_annotations(stimulus_ids):
    anno_file = "/home/ytang/workspace/data/mcmahon/social_dyad_videos_500ms/annotations.csv"
    import pandas as pd
    df = pd.read_csv(anno_file)
    
    annos = {}
    for id in stimulus_ids:
        anno = df[df.video_name==id].iloc[0].to_dict()
        anno.pop("video_name")
        annos[id] = anno
    return annos

def get_video_paths():
    video_dir = "/home/ytang/workspace/data/mcmahon/social_dyad_videos_500ms/"
    train_files = _get_files(os.path.join(video_dir, "train.csv"))
    test_files = _get_files(os.path.join(video_dir, "test.csv"))

    all_files = train_files + test_files
    all_paths = []
    for file in all_files:
        path = os.path.join(video_dir, "dyad_videos_3000ms", file)
        assert os.path.exists(path), f"{path} does not exist"
        all_paths.append(path)
    return all_files, all_paths
