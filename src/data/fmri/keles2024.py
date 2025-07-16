import os
from brainio.assemblies import NeuroidAssembly
from brainio.stimuli import StimulusSet
import numpy as np
from joblib import Parallel, delayed
from brainscore_vision.model_helpers.activations.temporal.utils import stack_with_nan_padding
from brainscore_vision.metrics import Score
from brainscore_vision.metric_helpers.transformations import apply_aggregate, standard_error_of_the_mean


FMRI_PREP_OUTPUT_DIR = "/home/ytang/workspace/data/keles2024/brainscore/fmriprep"
MOVIE_DIR = "/home/ytang/workspace/data/keles2024"
SURFACE_MESH = "fsaverage5"
MOVIES = ["short.avi"]
NUM_MESH_VERTEX = 10242
TR = 1000


def _load_surface(path):
    from nilearn import surface
    return surface.load_surf_data(path).astype(float)

def _chop_surface_time(surf):
    return surf[..., 10:10+478]  # roughly 10s - 488s

def _load_confounds(path):
    import pandas as pd
    df = pd.read_csv(path, sep="\t")
    df = df.fillna(df.mean())
    return df.values.T

def _regress_out_confounds(data, confounds):
    from sklearn.decomposition import PCA
    Y = data.T
    X = confounds.T
    X = X - X.mean(axis=0)
    X = X / (X.std(axis=0)+1e-8)

    pca = PCA(n_components=.95, whiten=True)
    pca.fit(X)
    print(f"PCA: {pca.n_components_}/{X.shape[1]} components")
    X = pca.transform(X)
    Y_hat = np.linalg.lstsq(X, Y, rcond=None)[0]
    Y_res = Y - X @ Y_hat
    return Y_res.T

def get_surfaces():
    from nilearn import image
    import numpy as np
    from matplotlib import pyplot as plt

    from nilearn import plotting
    import nibabel as nib
    import re

    # example filename: "sub-p42cs_ses-001_task-movie_run-1_space-fsaverage5_hemi-L_bold.func.gii"
    # confound filenames: sub-p41cs_ses-001_task-movie_run-1_desc-confounds_timeseries.tsv

    surfaces = []
    confounds = []
    for dirpath, dirnames, filenames in os.walk(FMRI_PREP_OUTPUT_DIR):
        for dirname in dirnames:
            if dirname.startswith("sub-"):
                for run in [1,2]:
                    surf_l_path = os.path.join(dirpath, dirname, "ses-001/func", f"{dirname}_ses-001_task-movie_run-{run}_space-{SURFACE_MESH}_hemi-L_bold.func.gii")
                    surf_r_path = os.path.join(dirpath, dirname, "ses-001/func", f"{dirname}_ses-001_task-movie_run-{run}_space-{SURFACE_MESH}_hemi-R_bold.func.gii")
                    confound_path = os.path.join(dirpath, dirname, "ses-001/func", f"{dirname}_ses-001_task-movie_run-{run}_desc-confounds_timeseries.tsv")
                    surf_l = _load_surface(surf_l_path)
                    surf_r = _load_surface(surf_r_path)
                    surf = np.concatenate([surf_l, surf_r], axis=0)
                    if surf.shape[-1] != 500:
                        print(surf.shape[-1], surf_l_path)
                        # surf = surf[...,-500:]
                        continue
                    confound = _load_confounds(confound_path)
                    surf = _regress_out_confounds(surf, confound)
                    surf = _chop_surface_time(surf)
                    surfaces.append(surf)
                    confounds.append(confound)

    print(f"Loaded {len(surfaces)} surfaces.")

    data = np.stack(surfaces)

    return data


def load_dataset(identifier='Keles2024-fMRI'):

    stimulus_ids = MOVIES
    stimulus_paths = [os.path.join(MOVIE_DIR, movie) for movie in MOVIES]
    data = get_surfaces()
    NUM_MESH_VERTEX = data.shape[1] // 2
    R, N, T = data.shape

    assembly = NeuroidAssembly(
        data[None, ...],
        dims=('stimulus_id', 'repetition', 'neuroid', 'time_bin'),
        coords={
            'repetition': ("repetition", np.arange(R)),
            'neuroid_id': ("neuroid", np.arange(N)),
            'hemi': ("neuroid", np.array(["L"]*NUM_MESH_VERTEX + ["R"]*NUM_MESH_VERTEX)),
            'time_bin_start': ("time_bin", (np.arange(T))*TR),
            'time_bin_end': ("time_bin", (1+np.arange(T))*TR),
            'stimulus_id': ("stimulus_id", stimulus_ids),
        },
    )

    assembly = assembly.stack(presentation=("stimulus_id", "repetition"))

    # make stimulus set
    stimulus_set = StimulusSet([{'stimulus_path': path, "stimulus_id": i} 
                                for i, path in zip(stimulus_ids, stimulus_paths)])
    stimulus_set.identifier = f'Keles2024-fMRI'
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly