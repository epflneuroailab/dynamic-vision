import os
from brainio.assemblies import NeuronRecordingAssembly
from brainio.stimuli import StimulusSet

import xarray as xr
import numpy as np
from brainscore_vision.model_helpers.brain_transformation.temporal import assembly_time_align
from brainscore_vision.model_helpers.activations.temporal.inputs import Video
from PIL import Image


root = "/home/ytang/workspace/data/oleo"

def find_path_helper(root, name):
    for root, dirs, files in os.walk(root, topdown=True):
        if name in dirs or name in files:
            return os.path.join(root, name)

def find_path(root, name):
    """
        Find a (the first encountered) file/directory in the current 
        directory or recursively in the children directories.
        "name" can contain hierarchical path: e.g. "a/b/c.txt"
    """
    names = os.path.normpath(name).split(os.sep)
    scope = root
    for name in names:
        scope = find_path_helper(scope, name)
        if scope is None:
            raise FileNotFoundError(
                f"Could not find {name} in {root} or any of its children directories.")
    return scope


def _get_relative_path(filepath):
    return filepath.replace("/braintree/home/sachis/temporal_integration/monkey_experiments/", "")


def load_dataset(identifier):
    asm = load_dataset_temporal(identifier.replace("-static", "").replace("-temporal", ""))
    if identifier.endswith("-temporal"):
        return asm
    elif identifier.endswith("-static"):
        stimulus_set = asm.stimulus_set
        asm = assembly_time_align(asm, [(40, 140)]).squeeze('time_bin')

        asm.attrs["stimulus_set"] = stimulus_set
        asm.attrs["stimulus_set_identifier"] = identifier
        asm.stimulus_set.identifier = identifier

        out_dir = os.path.join(root, "static")
        os.makedirs(out_dir, exist_ok=True)
        new_paths = {}
        for stimulus_id, stimulus_path in asm.stimulus_set.stimulus_paths.items():
            vid = Video.from_path(stimulus_path)
            first_frame = vid.get_frames([0])[0]
            img = Image.fromarray(first_frame)
            img_path = os.path.join(out_dir, f"{stimulus_id}.png")
            img.save(img_path)
            new_paths[stimulus_id] = img_path

        asm.stimulus_set.stimulus_paths = new_paths
        return asm

def load_dataset_temporal(identifier, time_start=200, time_end=700):
    assembly_class = NeuronRecordingAssembly

    if identifier == "oleo.hacs250-pclips":
        filename = "oleo.rsvp.hacs250-pclips.experiment_psth.nc"
    elif identifier == "oleo.hacs50-pclips":
        filename = "oleo.rsvp.hacs50-pclips.experiment_psth.nc"
    elif identifier == "oleo.hacs50-pclips-corrupted":
        filename = "oleo.rsvp.hacs50-pclips-corrupted.experiment_psth.nc"
    elif identifier == "oleo.hacs50-frames":
        filename = "oleo.rsvp.hacs50-frames.experiment_psth.nc"

    input_path = find_path(root, filename)
    arr = xr.open_dataarray(input_path)

    # video_identifier -> stimulus_id
    # time_bin_stop -> time_bin_end
    # true_label(class) -> true_label(class)
    if 'video_identifier' in arr.coords:
        arr = arr.rename(video_identifier="stimulus_id")
    else:
        if "stimulus_id" in arr.coords:
            arr = arr.rename(stimulus_id="stimulus_id_")
        arr = arr.rename(image_id="stimulus_id")
    if 'corruption_type' in arr.coords:
        # add corruption_type to stimulus_id
        arr = arr.assign_coords(stimulus_id=('presentation', [f"{sid}-{ctype}" for sid, ctype in zip(arr.stimulus_id.values, arr.corruption_type.values)]))
    arr = arr.rename(video_label="true_label")
    arr = arr.rename(video_class="true_class")

    arr = assembly_class(arr)  # bug in xr: you have to do this so that unloaded coords are not turned to variables

    paths = arr.image_current_local_file_path.values
    paths, idx = np.unique(paths, return_index=True)
    paths = [find_path(root, _get_relative_path(path)) for path in paths]
    stimulus_set = {}
    stimulus_set["stimulus_id"] = arr["stimulus_id"].values[idx]
    stimulus_set["true_label"] = arr["true_label"].values[idx]
    stimulus_set = StimulusSet(stimulus_set)
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], paths)}
    stimulus_set.identifier = filename.replace(".nc", "").replace(".experiment_psth", "")

    # attach
    arr.attrs["stimulus_set"] = stimulus_set
    arr.attrs["stimulus_set_identifier"] = stimulus_set.identifier

    # clip time
    arr = arr.isel(time_bin=arr.time_bin_start >= time_start)
    arr = arr.isel(time_bin=arr.time_bin_end < time_end)

    return arr