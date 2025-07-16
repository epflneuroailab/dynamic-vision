import os
from brainio.assemblies import NeuroidAssembly
from brainio.stimuli import StimulusSet
import numpy as np
from joblib import Parallel, delayed
from brainscore_vision.model_helpers.activations.temporal.utils import stack_with_nan_padding
from brainscore_vision.metrics import Score
from brainscore_vision.metric_helpers.transformations import apply_aggregate, standard_error_of_the_mean


FMRI_DIR = "/home/ytang/workspace/data/mcmahon/betas"
SURFACE_MESH = "fsaverage5"
NUM_MESH_VERTEX = 10_242


def map_to_surface(volume, radius=3):
    # radius: mm
    from nilearn import surface
    from nilearn import datasets

    fsaverage = datasets.fetch_surf_fsaverage(SURFACE_MESH)
    surface_l = surface.vol_to_surf(volume, fsaverage.pial_left, radius=radius)  # [F, T]
    surface_r = surface.vol_to_surf(volume, fsaverage.pial_right, radius=radius)
    surface = np.concatenate([surface_l, surface_r], axis=0)  # [2 * NUM_MESH_VERTEX(l,r), T]
    return surface

def get_paths():
    import re

    # example: sub-04_space-T1w_desc-test-fracridge(_odd/_even/"empty")_data_MNI152NLin2009cAsym.nii.g
    pattern = re.compile(r"sub-(\d+)_space-T1w_desc-(train|test)-fracridge(-odd|-even|)_data_MNI152NLin2009cAsym.nii.gz")

    subjs = []
    paths = []
    for dirpath, dirnames, filenames in os.walk(FMRI_DIR):
        for filename in filenames:
            match = pattern.match(filename)
            if match is not None:
                subj = int(match.group(1))
                mode = match.group(2)
                run = match.group(3)
                if run: 
                    run = run[1:]
                else:
                    run = "all"
                subjs.append((subj, mode, run))
                paths.append(os.path.join(dirpath, filename))
    return subjs, paths


def make_surfaces():
    from nilearn import image
    import numpy as np
    from matplotlib import pyplot as plt

    from nilearn import plotting
    import nibabel as nib
    import re

    subjs, paths = get_paths()
    N_SUBJ = len(subjs)

    _set_NUM_MESH_VERTEX()
    print(f"NUM_MESH_VERTEX: {NUM_MESH_VERTEX}")

    # batching in 8
    for i in range(0, N_SUBJ, 8):
        print(f"{i}/{N_SUBJ}")
        batch = paths[i:i+8]
        imgs = [image.load_img(subj) for subj in batch]

        # surfaces = [map_to_surface(img) for img in imgs]
        surfaces = Parallel(n_jobs=-1)(delayed(map_to_surface)(img) for img in imgs)

        # store
        for surface, subj in zip(surfaces, batch):
            path = subj.replace(".nii.gz", "_surface.npy")
            np.save(path, surface)


def _load_surface(path):
    return np.load(path)

def _get_N(path):
    import nibabel as nib
    if "test" in path:
        return 50
    else:
        return 200

def get_surfaces():
    from nilearn import image
    import numpy as np

    from nilearn import plotting
    import nibabel as nib
    import re

    subjs, paths = get_paths()
    paths = [path.replace(".nii.gz", "_surface.npy") for path in paths]

    data_train = []
    data_test = []
    data_train_even = []
    data_test_even = []
    data_train_odd = []
    data_test_odd = []
    for i, (subj, path) in enumerate(zip(subjs, paths)):
        print(f"{i+1}/{len(paths)}")
        subj, mode, run = subj
        d = _load_surface(path)
        
        if mode == "train":
            if run == "all":
                data_train.append(d)
            elif run == "even":
                data_train_even.append(d)
            elif run == "odd":
                data_train_odd.append(d)
        elif mode == "test":
            if run == "all":
                data_test.append(d)
            elif run == "even":
                data_test_even.append(d)
            elif run == "odd":
                data_test_odd.append(d)

    data_train = np.array(data_train)
    data_test = np.array(data_test)
    data_train_even = np.array(data_train_even)
    data_test_even = np.array(data_test_even)
    data_train_odd = np.array(data_train_odd)
    data_test_odd = np.array(data_test_odd)

    data = np.concatenate([data_train, data_test], axis=-1)
    data_even = np.concatenate([data_train_even, data_test_even], axis=-1)
    data_odd = np.concatenate([data_train_odd, data_test_odd], axis=-1)

    # merge subjects into repetitions
    data = np.concatenate([data_even, data_odd], axis=0)  # 8 x N x P=250
    return data

def load_stimulus_set(identifier='McMahon2023-fMRI'):
    stimulus_ids, stimulus_paths = get_video_paths()
    annos = get_annotations(stimulus_ids)
    stimulus_set = StimulusSet([{'stimulus_path': path, "stimulus_id": i, **annos[i]} 
                                for i, path in zip(stimulus_ids, stimulus_paths)])
    stimulus_set.identifier = f'McMahon2023-fMRI'
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}
    return stimulus_set, stimulus_ids


def load_dataset(identifier='McMahon2023-fMRI'):
    stimulus_set, stimulus_ids = load_stimulus_set(identifier)
    data = get_surfaces()
    R, N, P = data.shape

    assembly = NeuroidAssembly(
        data[None, ...],
        dims=('time_bin', 'repetition', 'neuroid', 'stimulus_id'),
        coords={
            'neuroid_id': ("neuroid", np.arange(N)),
            'hemi': ("neuroid", np.array(["L"]*NUM_MESH_VERTEX + ["R"]*NUM_MESH_VERTEX)),
            'stimulus_id': ("stimulus_id", stimulus_ids),
            'repetition': ("repetition", np.arange(R)),
            'time_bin_start': ("time_bin", [0]),
            'time_bin_end': ("time_bin", [3000]),
        },
    )

    assembly = assembly.stack(presentation=("stimulus_id", "repetition"))

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
