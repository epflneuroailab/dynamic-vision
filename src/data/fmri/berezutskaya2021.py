import os
from brainio.assemblies import NeuroidAssembly
from brainio.stimuli import StimulusSet
import numpy as np
from joblib import Parallel, delayed
from brainscore_vision.model_helpers.activations.temporal.utils import stack_with_nan_padding
from brainscore_vision.metrics import Score
from brainscore_vision.metric_helpers.transformations import apply_aggregate, standard_error_of_the_mean
from brainscore_vision.model_helpers.brain_transformation.temporal import assembly_time_align


FMRI_PREP_OUTPUT_DIR = "/home/ytang/workspace/data/berezutskaya_betas/brainscore/fmriprep"
MOVIE_DIR = "/home/ytang/workspace/data/berezutskaya/stimuli"
MOVIES = ["pippi_on_the_run_edited4brain_recordings.avi"]
SURFACE_MESH = "fsaverage5"
NUM_MESH_VERTEX = 10242
TR = 608
TARGET_TR = 1000


def _load_surface(path):
    from nilearn import surface
    return surface.load_surf_data(path).astype(float)


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

    # example filename: "sub-04/ses-mri3t/func/sub-04_ses-mri3t_task-film_run-1_space-fsaverage5_hemi-L_bold.func.gii"
    # confound filename: "sub-04_ses-mri3t_task-film_run-1_desc-confounds_timeseries.tsv"

    surfaces = []
    for dirpath, dirnames, filenames in os.walk(FMRI_PREP_OUTPUT_DIR):
        for dirname in dirnames:
            if dirname.startswith("sub-"):
                for run in [1]:
                    surf_l_path = os.path.join(dirpath, dirname, "ses-mri3t/func", f"{dirname}_ses-mri3t_task-film_run-{run}_space-{SURFACE_MESH}_hemi-L_bold.func.gii")
                    surf_r_path = os.path.join(dirpath, dirname, "ses-mri3t/func", f"{dirname}_ses-mri3t_task-film_run-{run}_space-{SURFACE_MESH}_hemi-R_bold.func.gii")
                    confound_path = os.path.join(dirpath, dirname, "ses-mri3t/func", f"{dirname}_ses-mri3t_task-film_run-{run}_desc-confounds_timeseries.tsv")

                    if not (os.path.exists(surf_l_path) and os.path.exists(surf_r_path)):
                        print(f"Missing {dirname}")
                        continue

                    surf_l = _load_surface(surf_l_path)
                    surf_r = _load_surface(surf_r_path)
                    surf = np.concatenate([surf_l, surf_r], axis=0)
                    confound = _load_confounds(confound_path)
                    surf = _regress_out_confounds(surf, confound)
                    surfaces.append(surf)

    data = np.stack(surfaces)
    return data


def load_dataset(identifier='Berezutskaya2021-fMRI'):

    stimulus_ids = MOVIES
    stimulus_paths = [os.path.join(MOVIE_DIR, movie) for movie in MOVIES]
    data = get_surfaces()

    R, N, T = data.shape
    NUM_MESH_VERTEX = N // 2

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

    max_time = assembly['time_bin_end'].max().item()
    target_time_bins = [(i, i+TARGET_TR) for i in list(range(0, max_time, TARGET_TR))[:-1]]
    assembly = assembly_time_align(assembly, target_time_bins, mode="portion")

    # make stimulus set
    stimulus_set = StimulusSet([{'stimulus_path': path, "stimulus_id": i} 
                                for i, path in zip(stimulus_ids, stimulus_paths)])
    stimulus_set.identifier = f'Berezutskaya2021-fMRI'
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly
