import os
import pickle
from brainio.assemblies import NeuroidAssembly
from brainio.stimuli import StimulusSet
import numpy as np
from joblib import Parallel, delayed
from brainscore_vision.model_helpers.activations.temporal.utils import stack_with_nan_padding
from brainscore_vision.metrics import Score
from brainscore_vision.metric_helpers.transformations import apply_aggregate, standard_error_of_the_mean

from neuroparc.atlas import Atlas


FMRI_DIR = "/home/ytang/workspace/data/lahner2024/derivatives/versionB/fsaverage/GLM"
VIDEO_DIR = "/home/ytang/workspace/data/lahner2024/stimuli"
SURFACE_MESH = "fsaverage5"
NUM_MESH_VERTEX = 10_242


def _fsaverage7_to_5(data): 
    return Atlas("fsaverage7", data).label_surface(SURFACE_MESH)

def load_surface(paths):
    test_l, test_r, train_l, train_r = paths
    data = []
    vids = []
    for path in paths:
        with open(path, 'rb') as f:
            d, videos = pickle.load(f)
            videos = [v.replace("vid", "") for v in videos]
            d = d.mean(1)  # average over repetitions for this single subject
            data.append(d)
            vids.append(videos)

    assert vids[0] == vids[1]
    assert vids[2] == vids[3]
    test = np.concatenate(data[:2], axis=-1)
    train = np.concatenate(data[2:], axis=-1)
    surface = np.concatenate([test, train], axis=0)

    from joblib import Parallel, delayed

    def fsaverage7_to_5(i, s):
        print(f"Subject {i+1}/{len(surface)}", end="\r")
        return _fsaverage7_to_5(s)

    surface = Parallel(n_jobs=-1)(delayed(fsaverage7_to_5)(i, s) for i, s in enumerate(surface))
    surface = np.array(surface)
    videos = vids[0] + vids[2]
    return surface, videos

def get_paths():
    # example: sub-01_organized_betas_task-test_hemi-left_normalized.pkl
    pattern = "sub-{:02d}/prepared_betas/sub-{:02d}_organized_betas_task-{}_hemi-{}_normalized.pkl"

    paths = {}
    for sub in range(1, 11):
        paths[sub] = []
        for task in ["test", "train"]:
            for hemi in ["left", "right"]:
                path = os.path.join(FMRI_DIR, pattern.format(sub, sub, task, hemi))
                paths[sub].append(path)
    return paths


def make_surfaces():
    from nilearn import image
    import numpy as np
    from matplotlib import pyplot as plt

    from nilearn import plotting
    import nibabel as nib
    import re

    paths = get_paths()
    N_SUBJ = len(paths)

    print(f"NUM_MESH_VERTEX: {NUM_MESH_VERTEX}")

    videos = None
    surfaces = []
    for i, (subj, paths) in enumerate(paths.items()):
        print(f"{i+1}/{N_SUBJ}")
        subj = f"sub-{subj:02d}"
        surface, vids = load_surface(paths)
        surfaces.append(surface)
        if videos is not None:
            assert videos == vids
        else:
            videos = vids

    surfaces = np.array(surfaces)
    return surfaces, videos


def load_dataset(identifier='Lahner2024-fMRI'):
    data, stimulus_ids = make_surfaces()
    invalid = invalid_files()
    valid_index = [i for i, vid in enumerate(stimulus_ids) if vid not in invalid]
    R, P, N = data.shape

    assembly = NeuroidAssembly(
        data[None, ...],
        dims=('time_bin', 'repetition', 'stimulus_id', 'neuroid'),
        coords={
            'neuroid_id': ("neuroid", np.arange(N)),
            'hemi': ("neuroid", np.array(["L"]*NUM_MESH_VERTEX + ["R"]*NUM_MESH_VERTEX)),
            'stimulus_id': ("stimulus_id", stimulus_ids),
            'repetition': ("repetition", np.arange(R)),
            'time_bin_start': ("time_bin", [0]),
            'time_bin_end': ("time_bin", [3000]),
        },
    )

    assembly = assembly.isel(stimulus_id=valid_index)
    assembly = assembly.stack(presentation=("stimulus_id", "repetition"))

    # make stimulus set
    stimulus_ids = [stimulus_ids[i] for i in valid_index]
    stimulus_paths = [os.path.join(VIDEO_DIR, f"{vid}.mp4") for vid in stimulus_ids]
    stimulus_set = StimulusSet([{'stimulus_path': path, "stimulus_id": i} 
                                for i, path in zip(stimulus_ids, stimulus_paths)])
    stimulus_set.identifier = f'Lahner2024-fMRI'
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly


def invalid_files():
    return [
        "0106",
        "0378",
        "0026",
        "0184",
        "1021",
        "0363",
        "0734",
        "0540",
        "0459",
        "0550",
        "0293",
        "0998",
        "0786",
        "0506",
        "0701",
        "0436",
        "0224",
        "0216",
        "0040",
        "0172",
        "0032",
        "0737",
        "0914",
        "0201",
        "0473",
        "0333",
        "0976",
        "0597",
        "0237",
        "1020",
        "0494",
        "0839",
        "1031",
        "0028",
        "0064",
        "0990",
        "1015",
        "0347",
        "0365",
        "0894",
        "0542",
        "0666",
        "1060",
        "0104",
        "1044",
        "0809",
        "0156",
        "0343",
        "0890",
        "1019",
        "0978",
        "0460",
        "0027",
        "0063",
        "1004",
        "0919",
        "0496",
        "0989",
        "0276",
        "0710",
        "0982",
        "0779",
        "0992",
        "0198",
        "0235",
        "0041",
        "0210",
        "0951",
        "0202",
        "0139",
        "0122",
        "0651",
        "0448",
        "1018",
        "0959",
        "0995",
    ]