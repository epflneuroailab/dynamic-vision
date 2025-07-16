import os
import numpy as np
import h5py
from brainio.stimuli import StimulusSet
from brainio.assemblies import DataAssembly
from brainscore_vision.model_helpers.activations.temporal.inputs import Video
from result_caching import store


DATA_DIR = '/home/ytang/workspace/data/kinetics400/kinetics400'
MAX_CONDITIONS = 100
MAX_VIDEO_PER_CONDITION = 50
VIDEO_DURATION = 4000  # ms
VIDEO_FPS = 25


def load_dataset(identifier="kinetics400"):
    static = False
    if identifier.endswith('-static'):
        static = True
    stimulus_paths, stimulus_ids, labels = get_data(static)

    stimulus_set = {}
    stimulus_set["stimulus_id"] = stimulus_ids
    stimulus_set["action"] = labels
    stimulus_set = StimulusSet(stimulus_set)
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}
    stimulus_set.identifier = identifier

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


def get_data(static=False):
    np.random.seed(42)
    video_dir = os.path.join(DATA_DIR, 'test')
    out_dir = os.path.join(DATA_DIR, 'brainscore')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    stimulus_paths = []
    stimulus_ids = []
    labels = []

    folders = os.listdir(video_dir)
    np.random.shuffle(folders)
    valid_count = 0

    for i, action_type in enumerate(folders):
        if valid_count == MAX_CONDITIONS:
            break
        print(f"{i+1}/{len(folders)} {action_type}", end='\r')
        action_dir = os.path.join(video_dir, action_type)
        out_action_dir = os.path.join(out_dir, action_type)
        if not os.path.exists(out_action_dir):
            os.makedirs(out_action_dir)

        count = 0 
        stimulus_paths_to_add = []
        stimulus_ids_to_add = []
        labels_to_add = []
        for j, video_name in enumerate(os.listdir(action_dir)):
            if count == MAX_VIDEO_PER_CONDITION:
                break
            video_path = os.path.join(action_dir, video_name)

            try:
                video = Video.from_path(video_path)
                video_duration = video.duration
                if video.duration < VIDEO_DURATION:
                    print(f"Skip {video_name} due to short duration {video_duration}")
                    continue

                start = np.random.rand() * (video_duration - VIDEO_DURATION) if video_duration > VIDEO_DURATION else 0
                end = start + VIDEO_DURATION

                out_video_path = os.path.join(out_action_dir, video_name)
                video = video.set_fps(VIDEO_FPS).set_window(start, end)


                if static:
                    frames = video.to_frames()
                    frame = frames[len(frames)//2]
                    out_video_path = out_video_path.replace('.mp4', '.png')
                    write_frame(out_video_path, frame)
                else:
                    video.store_to_path(out_video_path)

            except Exception as e:
                print(f"Skip {video_name} due to error {e}")
                continue

            stimulus_paths_to_add.append(out_video_path)
            stimulus_ids_to_add.append(video_name)
            labels_to_add.append(action_type)
            count += 1

        if count == MAX_VIDEO_PER_CONDITION:
            valid_count += 1
            stimulus_paths.extend(stimulus_paths_to_add)
            stimulus_ids.extend(stimulus_ids_to_add)
            labels.extend(labels_to_add)
        else:
            import shutil
            shutil.rmtree(out_action_dir)

    labels = np.array(labels)
    return stimulus_paths, stimulus_ids, labels


def write_video(path, video, fps=10):
    # write to mp4; video: np.array
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (video.shape[2], video.shape[1]))
    for frame in video:
        out.write(frame)
    out.release()


def write_frame(path, frame):
    import cv2
    cv2.imwrite(path, frame[:, :, ::-1])