import os
import numpy as np
import h5py
from brainio.stimuli import StimulusSet
from brainio.assemblies import DataAssembly
from brainscore_vision.model_helpers.activations.temporal.inputs.video import get_video_stats
from result_caching import store


DATA_DIR = '/home/ytang/workspace/data/AFD/'
MAX_CONDITION = 101
MAX_VIDEO_PER_CONDITION = 50
VIDEO_DURATION = 4000  # ms
TOL_LOW = 1000  # ms
TOL_HIGH = 4000  # ms
FPS = 30


def load_dataset(identifier="afd101"):
    stimulus_paths, stimulus_ids, labels = get_data()

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


def get_data():
    np.random.seed(42)
    video_dir = os.path.join(DATA_DIR, 'afd101')
    out_dir = os.path.join(DATA_DIR, 'brainscore')
    os.makedirs(out_dir, exist_ok=True)
    stimulus_paths = []
    stimulus_ids = []
    labels = []

    video_conditions = os.listdir(video_dir)
    video_conditions = np.random.permutation(video_conditions)

    condition_count = 0
    for i, action_type in enumerate(video_conditions):
        if condition_count == MAX_CONDITION:
            break
        print(f"{i+1}/{len(os.listdir(video_dir))} {action_type}", end='\r')
        action_dir = os.path.join(video_dir, action_type)
        os.makedirs(os.path.join(out_dir, action_type), exist_ok=True)

        stimulus_paths_to_add = []
        stimulus_ids_to_add = []
        labels_to_add = []

        count = 0 
        actions = os.listdir(action_dir)
        actions = np.random.permutation(actions)
        for j, video_name in enumerate(actions):
            if count == MAX_VIDEO_PER_CONDITION:
                break
            video_path = os.path.join(action_dir, video_name)

            # try:
            fps, video_duration, size = get_video_stats(video_path)
            if not (VIDEO_DURATION+TOL_HIGH >= video_duration >= VIDEO_DURATION-TOL_LOW): 
                # print(f"Video duration is {video_duration} ms")
                continue

            num_frames = int(FPS * VIDEO_DURATION / 1000)

            frames = read_video(video_path)

            if frames.shape[0] != num_frames:
                if frames.shape[0] > num_frames:
                    start_frame = np.random.randint(0, frames.shape[0]-num_frames)
                    frames = frames[start_frame:start_frame+num_frames]
                else:
                    # extend the last frame
                    last_frame = frames[-1]
                    num_pad_frames = num_frames - frames.shape[0]
                    pad_frames = np.tile(last_frame, (num_pad_frames, 1, 1, 1))
                    frames = np.concatenate([frames, pad_frames], axis=0)

            store_path = os.path.join(out_dir, action_type, video_name.replace('.avi', '.avi'))
            write_video(store_path, frames, FPS)
            fps, video_duration, size = get_video_stats(store_path)
            assert video_duration == VIDEO_DURATION

            # except Exception as e:
            #     print(f"Skip {video_name} due to error {e}")
            #     continue

            stimulus_paths_to_add.append(store_path)
            stimulus_ids_to_add.append(video_name)
            labels_to_add.append(action_type)
            count += 1

        if count == MAX_VIDEO_PER_CONDITION:
            stimulus_paths.extend(stimulus_paths_to_add)
            stimulus_ids.extend(stimulus_ids_to_add)
            labels.extend(labels_to_add)
            condition_count += 1
        else:
            import shutil
            shutil.rmtree(os.path.join(out_dir, action_type)) 
            print(f"Skip {action_type} due to insufficient videos")
    
    labels = np.array(labels)
    return stimulus_paths, stimulus_ids, labels


def write_video(path, video, fps):
    # write to mp4; video: np.array
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (video.shape[2], video.shape[1]))
    for frame in video:
        out.write(frame)
    out.release()


def read_video(path):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)