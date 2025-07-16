import os
import numpy as np
from mat73 import loadmat
from brainio.assemblies import NeuroidAssembly
from brainio.stimuli import StimulusSet
from brainscore_vision.model_helpers.brain_transformation.temporal import assembly_time_align


DATA_DIR = "/home/ytang/workspace/data/yourhead/crcns-mt2"
output_dir = os.path.join(DATA_DIR, 'videos')
os.makedirs(output_dir, exist_ok=True)
FPS = 72
TARGET_TIME_BIN = 200  # ms
ORIGIN_TIME_BIN = 1000 / FPS  # ms
TARGET_NUM_FRAMES = 22000  # frames
MAX_VIDEO_TIME = 300000  # ms

count = 0

def get_single_recording(fname):
    path = os.path.join(DATA_DIR, fname)
    data = loadmat(path)
    frs = data['psths']
    stimulus = data['rawStims']
    cell = data['cellid']

    # fill mean to nan
    mean = np.nanmean(frs)
    frs[np.isnan(frs)] = mean

    stimulus = stimulus.transpose(2, 0, 1)
    return cell, frs, stimulus


def get_data(identifier="crcns-mt2"):
    global count
    cell_ids = []
    all_frs = []
    vid_paths = []
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith('.mat'):
            continue

        try:
            cellid, frs, stimulus = get_single_recording(fname)
            print(cellid)
        except:
            continue

        # filter target duration
        stim = stimulus[:TARGET_NUM_FRAMES]
        fires = frs[:TARGET_NUM_FRAMES]
        
        cell_ids.append(cellid)
        all_frs.append(fires)
        store_path = os.path.join(output_dir, f"{cellid}.avi")
        write_video(store_path, stim, FPS)
        vid_paths.append(store_path)

        count += 1

    all_frs = np.array(all_frs)
    return all_frs, vid_paths, cell_ids


def load_dataset(identifier="crcns-mt2"):
    all_frs, vid_paths, cell_ids = get_data()
    print(all_frs.shape)
    stimulus_paths = vid_paths

    P = len(all_frs)
    N = P
    T = len(all_frs[0])
    duration = T * ORIGIN_TIME_BIN
    stimulus_ids = np.arange(P)

    frs = np.zeros((P, N, T)) * np.nan
    frs # presentation x neuroid x time
    for i, fr in enumerate(all_frs):
        frs[i, i, :] = fr.reshape(-1)

    assembly = NeuroidAssembly(
        frs,
        dims=('stimulus_id', 'neuroid', 'time_bin'),
        coords={
            'neuroid_id': ("neuroid", np.arange(N)),
            'cell_id': ("neuroid", cell_ids),
            'time_bin_start': ("time_bin", (np.arange(T))*ORIGIN_TIME_BIN),
            'time_bin_end': ("time_bin", (1+np.arange(T))*ORIGIN_TIME_BIN),
            'stimulus_id': ("stimulus_id", stimulus_ids),
        },
    )

    target_time_starts = np.arange(0, MAX_VIDEO_TIME, TARGET_TIME_BIN)
    target_time_ends = target_time_starts + TARGET_TIME_BIN
    target_time_bins = [(start, end) for start, end in zip(target_time_starts, target_time_ends)]
    assembly = assembly_time_align(assembly, target_time_bins, mode="portion")

    assembly = assembly.assign_coords(video=('stimulus_id', [os.path.basename(path) for path in vid_paths]))
    assembly = assembly.stack(presentation=("stimulus_id", ))
    assembly = assembly.__class__(assembly)

    # make stimulus set
    stimulus_set = StimulusSet([{'stimulus_path': path, "stimulus_id": i} 
                                for i, path in zip(stimulus_ids, stimulus_paths)])
    stimulus_set.identifier = f'crcns-mt2'
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly


def write_video(path, video, fps):
    # write to mp4; video: np.array
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (video.shape[2], video.shape[1]))
    for frame in video:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    out.release()