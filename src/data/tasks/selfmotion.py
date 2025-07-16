import os
import numpy as np
import h5py
from brainio.stimuli import StimulusSet
from brainio.assemblies import DataAssembly


DATA_DIR = '/home/ytang/workspace/data/yourhead/batch2'
MAX_VIDEO_PER_CONDITION = 500


def load_dataset(identifier="selfmotion"):
    stimulus_paths, stimulus_ids, data = get_data()
    conditions = [id.split("_")[0] for id in stimulus_ids]

    stimulus_set = {}
    stimulus_set["stimulus_id"] = stimulus_ids
    for i, key in enumerate(
        ["rotation_pitch", "rotation_yaw", "heading_pitch", "heading_yaw", "pitch", "yaw", "speed"]
    ):
        stimulus_set[key] = data[:, i]
    stimulus_set = StimulusSet(stimulus_set)
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}
    stimulus_set.identifier = identifier

    assembly = DataAssembly(
        data,
        dims=('presentation', 'label'),
        coords={
            'stimulus_id': ('presentation', stimulus_ids),
            "condition": ("presentation", conditions),
            'label': ('label', ["rotation_pitch", "rotation_yaw", "heading_pitch", "heading_yaw", "pitch", "yaw", "speed"]),
        }
    )

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly 


def get_data():
    video_dir = os.path.join(DATA_DIR, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    stimulus_paths = []
    stimulus_ids = []
    data = []

    for condition in os.listdir(DATA_DIR):
        if condition == 'videos': continue
        batch = os.listdir(os.path.join(DATA_DIR, condition))[0]
        path = os.path.join(DATA_DIR, condition, batch, "output.h5")
        with h5py.File(path, 'r') as f:
            labels = f['labels']
            depths = f['depth']
            names = labels.dtype.names
            short_videos = f['short_videos']

            label_indices = [
                names.index('rotation_pitch'), 
                names.index('rotation_yaw'), 
                names.index('heading_pitch'),
                names.index('heading_yaw'),
                names.index('pitch'),
                names.index('yaw'),
                names.index('speed'),
            ]

            N = len(labels)
            for i in range(min(N, MAX_VIDEO_PER_CONDITION)):
                print(f"Processing {condition}_{batch}_{i}/{N}")
                video = short_videos[i]
                label = labels[i]
                label = np.array([label[i] for i in label_indices])
                path = os.path.join(video_dir, f"{condition}_{batch}_{i}.mp4")
                if not os.path.exists(path):
                    write_video(path, np.array(video.transpose(0,2,3,1)))

                stimulus_paths.append(path)
                stimulus_ids.append(f"{condition}_{batch}_{i}")
                data.append(label)

    data = np.array(data)
    return stimulus_paths, stimulus_ids, data


def write_video(path, video, fps=10):
    # write to mp4; video: np.array
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (video.shape[2], video.shape[1]))
    for frame in video:
        out.write(frame)
    out.release()