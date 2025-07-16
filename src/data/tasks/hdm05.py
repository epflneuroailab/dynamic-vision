import c3d
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import convolve

from brainio.stimuli import StimulusSet
from brainio.assemblies import DataAssembly
from brainscore_vision.model_helpers.activations.temporal.inputs.video import get_video_stats


c3d_path = "/home/ytang/workspace/data/hdm05/HDM05_cut_c3d"
out_dir = "/home/ytang/workspace/data/hdm05/HDM05_cut_avi"
height = 224
width = 224
noise_ratio = 0.02
dot_size = 3

MAX_VIDEO_PER_CONDITION = 50


np.random.seed(42)


def _find_first_digit(s):
    for i, c in enumerate(s):
        if c.isdigit():
            return i
    return -1

def find_class(raw_cls_name):
    pass

def write_frames(arrays, path):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height))
    for array in arrays:
        writer.write(array)
    writer.release()

def padding(frames, length):
    F = len(frames)
    if F >= length:
        return frames[:length]
    else:
        F_ = length // F + 1
        return np.concatenate([frames] * F_)[:length]

def process(class_name, filename, rotations=[0, 30, 60, 90]):
    path = f"{c3d_path}/{class_name}/{filename}"
    first_digit = _find_first_digit(class_name)
    agg_class_name = class_name[:first_digit] if first_digit != -1 else class_name
    dirname = f"{out_dir}/{agg_class_name}"
    os.makedirs(dirname, exist_ok=True)

    for rot in rotations:
        rot = rot % 360
        out_path = f"{dirname}/{class_name}-{rot}-{filename.replace('.C3D', '.avi')}"

        frames = []
        with open(path, 'rb') as handle:
            reader = c3d.Reader(handle)
            for frame, data, _ in reader.read_frames():
                frames.append(data)

        frames = padding(frames, 400)
        frames = np.array(frames)
        frames = frames[::4]  # 100 frames
        frames = frames[..., :3]

        frames = (frames - frames.reshape(-1, 3).min(0))  / (frames.reshape(-1, 3).max(0) - frames.reshape(-1, 3).min(0))

        r_xs = frames[:, :, 1]
        r_ys = frames[:, :, 2]
        r_zs = frames[:, :, 0]

        # random rotation along y axis
        ang = rot / 180 * np.pi
        c, s = np.cos(ang), np.sin(ang)
        r_xs, r_zs = c * r_xs - s * r_zs, s * r_xs + c * r_zs

        # move the camera to where the xs are
        r_x_min = r_xs.min()
        r_xs = r_xs - r_x_min

        xs = (r_xs * width).astype(int)
        ys = (r_ys * height).astype(int)
        xs = np.clip(xs, 0, width-1)
        ys = np.clip(ys, 0, height-1)

        vids = np.zeros((len(frames), height, width, 3), dtype=np.uint8)
        for i, (x, y) in enumerate(zip(xs, ys)):
            vid = np.zeros((height, width, 3), dtype=np.uint8)
            vid[y, x] = 255
            vids[i] = vid

        vids = np.flip(vids, 1)

        noise = np.random.rand(*vids.shape[1:-1])
        noise = (noise < noise_ratio) * 255

        vids = (vids + noise[None, ..., None]).astype(np.uint8)

        # convolve to make the dot bigger
        frames = [convolve(f, np.ones((dot_size, dot_size, 1)), mode='constant', cval=0) for f in vids]
        vids = np.array(frames)

        vids = np.clip(vids, 0, 255)

        write_frames(vids, out_path)
        print(out_path)


def get_videos():
    cache = []
    for class_name in os.listdir(c3d_path):
        if not os.path.isdir(f"{c3d_path}/{class_name}"):
            continue
        for filename in os.listdir(f"{c3d_path}/{class_name}"):
            cache.append((class_name, filename))

    from joblib import Parallel, delayed
    Parallel(n_jobs=-1)(delayed(process)(class_name, filename) for class_name, filename in cache)


def get_data():
    stimulus_paths = []
    stimulus_ids = []
    labels = []

    for class_name in os.listdir(out_dir):
        count = 0
        stimulus_paths_to_add = []
        stimulus_ids_to_add = []
        labels_to_add = []
        if not os.path.isdir(f"{out_dir}/{class_name}"):
            continue
        filenames = os.listdir(f"{out_dir}/{class_name}")
        # random shuffle
        filenames = np.random.permutation(filenames)
        for filename in filenames:
            stimulus_paths_to_add.append(f"{out_dir}/{class_name}/{filename}")
            stimulus_ids_to_add.append(filename)
            labels_to_add.append(class_name)

            count += 1
            if count == MAX_VIDEO_PER_CONDITION:
                break

        if count == MAX_VIDEO_PER_CONDITION:
            stimulus_paths.extend(stimulus_paths_to_add)
            stimulus_ids.extend(stimulus_ids_to_add)
            labels.extend(labels_to_add)
        else:
            print(f"Class {class_name} has only {count} videos, less than {MAX_VIDEO_PER_CONDITION}")

    return stimulus_paths, stimulus_ids, labels


def load_dataset(identifier="hdm05"):
    stimulus_paths, stimulus_ids, labels = get_data()

    stimulus_set = {}
    stimulus_set["stimulus_id"] = stimulus_ids
    stimulus_set["action"] = labels
    stimulus_set = StimulusSet(stimulus_set)
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}
    stimulus_set.identifier = identifier

    labels = np.array(labels)

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