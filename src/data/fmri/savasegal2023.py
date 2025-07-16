import os
from brainio.assemblies import NeuroidAssembly
from brainio.stimuli import StimulusSet
import numpy as np
from joblib import Parallel, delayed
from brainscore_vision.model_helpers.activations.temporal.utils import stack_with_nan_padding
from brainscore_vision.metrics import Score
from brainscore_vision.metric_helpers.transformations import apply_aggregate, standard_error_of_the_mean


MOVIE_DIR = "/home/ytang/workspace/data/event-segmentation/origin"
MOVIES = ["Defeat.mp4", "Growth.mp4", "Iteration.mp4", "Lemonade.mp4"]
FMRI_VOLUME_DIR = "/home/ytang/workspace/data/savasegal/volumes"
SURFACE_MESH = "fsaverage5"
NUM_MESH_VERTEX = 10_242
TR = 1000  # ms


"""
The films are padded so I recommend cutting 2 TRs off the front of each movie and 11 off the end of Growth, Lemonade and Iteration 
and 12 off the end of Defeat.
"""
def _load_chopped_surface(surface_path):
    surface = np.load(surface_path)
    if "defeat" in surface_path:
        surface = surface[..., 2:-12]
    else:
        surface = surface[..., 2:-11]
    return surface

def load_dataset(movie, identifier='SavaSegal2023-fMRI'):

    stimulus_ids = [movie]
    stimulus_paths = [os.path.join(MOVIE_DIR, movie)]
    data = get_surfaces(movie)

    R, N, T = data.shape
    durations = [T * TR for d in data]

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
    stimulus_set.identifier = f'SavaSegal-fMRI-{movie}'
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly


def map_to_surface(volume, radius=3):
    # radius: mm
    from nilearn import surface
    from nilearn import datasets

    fsaverage = datasets.fetch_surf_fsaverage(SURFACE_MESH)
    surface_l = surface.vol_to_surf(volume, fsaverage.pial_left, radius=radius)  # [F, T]
    surface_r = surface.vol_to_surf(volume, fsaverage.pial_right, radius=radius)
    surface = np.concatenate([surface_l, surface_r], axis=0)  # [2 * NUM_MESH_VERTEX(l,r), T]
    return surface


def make_surfaces(movie):
    from nilearn import image
    import numpy as np
    from matplotlib import pyplot as plt

    from nilearn import plotting
    import nibabel as nib
    import re

    pattern = re.compile(r"sub-NSD(\d+)_task-(\w+)_nocensor.nii.gz")

    subjs = []
    for filename in os.listdir(FMRI_VOLUME_DIR):
        match = pattern.match(filename)
        if match is not None:
            subj = match.group(1)
            task = match.group(2)
            if task != movie: continue
            subjs.append(subj)

    N_SUBJ = len(subjs)

    subjs = [
        os.path.join(FMRI_VOLUME_DIR, f'sub-NSD{subj}_task-{movie}_nocensor.nii.gz')
        for subj in subjs
    ]

    print(f"NUM_MESH_VERTEX: {NUM_MESH_VERTEX}")

    # batching in 8
    for i in range(0, N_SUBJ, 8):
        print(f"{i}/{N_SUBJ}")
        batch = subjs[i:i+8]
        imgs = [image.load_img(subj) for subj in batch]

        # check if has common affine
        af = None
        for img in imgs:
            if af is not None:
                assert (af == img.affine).all()
            else:
                af = img.affine

        # surfaces = [map_to_surface(img) for img in imgs]
        surfaces = Parallel(n_jobs=-1)(delayed(map_to_surface)(img) for img in imgs)

        # store
        for surface, subj in zip(surfaces, batch):
            path = subj.replace(".nii.gz", "_surface.npy")
            np.save(path, surface)


def _get_T(path):
    import nibabel as nib
    s = _load_chopped_surface(path)
    return s.shape[-1]

def get_surfaces(movie, make=False):
    from nilearn import image
    import numpy as np
    from matplotlib import pyplot as plt

    from nilearn import plotting
    import nibabel as nib
    import re

    # example filename: sub-NSD103_task-growth_nocensor.nii.gz
    if ".mp4" in movie:
        movie = movie.replace(".mp4", "")
    movie = movie.lower()
    if make:
        make_surfaces(movie)

    surfaces = []
    for filename in os.listdir(FMRI_VOLUME_DIR):
        if movie not in filename:
            continue
        if "surface" not in filename:
            continue
        surfaces.append(os.path.join(FMRI_VOLUME_DIR, filename))

    T = _get_T(surfaces[0])
    data = np.zeros((len(surfaces), 2 * NUM_MESH_VERTEX, T))  # R, N, T

    for i, surface in enumerate(surfaces):
        print(f"{i+1}/{len(surfaces)}")
        data[i] = _load_chopped_surface(surface)

    return data
    # data: [R, N, T]
    # 10 random splits along R


    def aggregate(values):
        score = apply_aggregate(lambda scores: scores.median('neuroid').mean('split'), values)
        score.attrs['error'] = standard_error_of_the_mean(values.median('neuroid'), 'split')
        return score

    R, N, T = data.shape
    scores = []
    np.random.seed(42)  # make sure splits for every movie are the same
    for _ in range(n_splits):
        idx = np.random.permutation(R)
        data1 = data[idx[:R//2]].mean(0)
        data2 = data[idx[R//2:]].mean(0)
        score_neuroids = []
        for n in range(N):
            d1 = data1[n]
            d2 = data2[n]
            c = np.corrcoef(d1, d2)[0, 1]
            score_neuroids.append(c)
        scores.append(score_neuroids)

    scores = np.array(scores)

    # Spearman-Brown correction
    scores = 2 * scores / (1 + scores)
    
    scores = Score(
        scores,
        dims=('split', 'neuroid'),
        coords={
            'neuroid_id': ("neuroid", np.arange(N)),
            'hemi': ("neuroid", np.array(["L"]*NUM_MESH_VERTEX + ["R"]*NUM_MESH_VERTEX)),
            'split': ("split", np.arange(n_splits)),
        },
    )

    score = aggregate(scores)
    return score