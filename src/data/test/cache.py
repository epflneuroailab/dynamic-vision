import os
import cv2
import numpy as np
from brainio.stimuli import StimulusSet
from brainio.assemblies import NeuroidAssembly

from .. import data_store


def make_fake_video(path, duration=60000, fps=25):
    num_frames = int(duration / 1000 * fps)
    frame_size = (224, 224)
    video = np.random.randint(0, 255, (num_frames, *frame_size, 3), dtype=np.uint8)

    cap = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    for frame in video:
        cap.write(frame)
    cap.release()

def make_fake_fmri(video_path, duration=60000, TR=1000):
    num_volumes = int(duration / TR)
    num_voxels = 10_242 * 2
    num_repeats = 4
    signal = np.random.randn(num_volumes, num_voxels)
    noise = np.random.randn(num_repeats, num_volumes, num_voxels) * 0.1
    data = signal[None, :, :] + noise

    assembly = NeuroidAssembly(
        data[None, ...],
        dims=('stimulus_id', 'repetition', 'time_bin', 'neuroid'),
        coords={
            'repetition': ("repetition", np.arange(num_repeats)),
            'neuroid_id': ("neuroid", np.arange(num_voxels)),
            'hemi': ("neuroid", ["L"] * (num_voxels//2) + ["R"] * (num_voxels//2)),
            'time_bin_start': ("time_bin", (np.arange(num_volumes)) * TR),
            'time_bin_end': ("time_bin", (1 + np.arange(num_volumes)) * TR),
            'stimulus_id': ("stimulus_id", ["fake_video.mp4"]),
        },
    )

    assembly = assembly.stack(presentation=("stimulus_id", "repetition"))

    stimulus_set = StimulusSet([{'stimulus_path': video_path, "stimulus_id": "fake_video.mp4"}])
    stimulus_set.identifier = 'Fake-fMRI'
    stimulus_set.stimulus_paths = {"fake_video.mp4": video_path}

    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly


video_path = os.path.join(data_store.root, "fake_video.mp4")
make_fake_video(video_path)
assembly = make_fake_fmri(video_path)
data_store.store(assembly, "fake-fmri")