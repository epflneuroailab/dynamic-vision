import os
import pandas as pd
import random
import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.stimuli import StimulusSet


CSV_FILE = "/home/ytang/workspace/data/han2024/qualified_human_data.csv"
VIDEO_HOME = "/work/upschrimpf1/akgokce/datasets/bmp_dataset/bmp_dataset_mp4"
FITTING_VIDEO_HOME = "/work/upschrimpf1/akgokce/datasets/bmp_dataset/bmp_dataset_fitting_mp4"

TYPES = [
    "J-26P",  
    "J-6P-0V",   
    "J-6P-4F",   
    "J-6P-S",  
    "RGB-4F",  
    "SP-4P-1LT",  
    "SP-8P-1LT",
    "J-5P",   
    "J-6P-3F",   
    "J-6P-90V",  
    "RGB",
    "RGB-R",   
    "SP-4P-2LT",  
    "SP-8P-2LT",
    "J-6P",   
    "J-6P-45V",  
    "J-6P-R",
    "RGB-3F",  
    "RGB-S",   
    "SP-4P-4LT",  
    "SP-8P-4LT",
]

TYPES_TO_INCLUDE = [
    "RGB",
    "RGB-4F",
    "RGB-S",
    "J-6P",
    "J-6P-4F",
    "J-6P-S",
]

ALL_PATHS = {}
for type in os.listdir(VIDEO_HOME):
    for file in os.listdir(os.path.join(VIDEO_HOME, type)):
        ALL_PATHS[(type, file)] = os.path.join(VIDEO_HOME, type, file)


def get_label_from_videoname(videoname):
    label = 'A'+videoname.split('.')[0].split('A')[-1].split('_')[0]
    return label

def load_dataset(identifier='han2024-RGB'):
    type = identifier.replace('han2024-', '')
    # assert type in ["RGB", "J", "SP"]

    df = pd.read_csv(CSV_FILE)
    df = df.iloc[[v == type for v in df['BMP Condition']]]

    stimulus_ids = df["Video Path"].values
    subjects = df["ID"].values
    truth_names = df["Ground Truth"].values
    truths = [get_label_from_videoname(v) for v in stimulus_ids]
    truth_name_map = {name: code for name, code in zip(truth_names, truths)}
    human_responses = df["Human Response"].values
    human_responses = [truth_name_map[name] for name in human_responses]
    conditions = df["BMP Condition"].values

    assembly = BehavioralAssembly(
        human_responses,
        dims=['presentation'],
        coords={
            "stimulus_id": ("presentation", stimulus_ids),
            "condition": ("presentation", conditions),
            "subject": ("presentation", subjects),
            "truth": ("presentation", truths),
            "truth_name": ("presentation", truth_names),
        }
    )

    # remove repeated stimuli
    df = df.drop_duplicates(subset=["Video Path"])
    stimulus_ids = df["Video Path"].values
    stimulus_paths = df["Video Path"].values
    subjects = df["ID"].values
    truth_names = df["Ground Truth"].values
    truths = [get_label_from_videoname(v) for v in stimulus_ids]

    stimulus_set = {}
    stimulus_set["stimulus_id"] = stimulus_ids
    stimulus_set['truth'] = truths
    stimulus_set['subject'] = subjects
    stimulus_set = StimulusSet(stimulus_set)
    stimulus_set.stimulus_paths = {id:os.path.join(VIDEO_HOME, type, path) for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}
    stimulus_set.identifier = identifier

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly


def load_stimulus_set(identifier='han2024-RGB-fitting_stimuli'):
    if identifier == 'han2024-RGB-fitting_stimuli':
        file_type = 'RGB'
    elif identifier == 'han2024-J-6P-fitting_stimuli':
        file_type = 'J-6P'

    # AlphaPose_S003C001P018R001A027_rgb -- A027 is the class
    fitting_video_dir = os.path.join(FITTING_VIDEO_HOME, file_type)

    data = []
    paths = {}
    for video_file in os.listdir(fitting_video_dir):
        cls = 'A'+video_file.split('.')[0].split('A')[-1].split('_')[0]
        paths[video_file] = os.path.join(fitting_video_dir, video_file)
        data.append({"stimulus_id": video_file, "truth": cls})

    print("Number of unique classes: ", len(np.unique([d['truth'] for d in data])))

    stimulus_set = StimulusSet(data)
    stimulus_set.stimulus_paths = paths
    stimulus_set.identifier = identifier
    return stimulus_set


# from ...evaluate.behaviour.i1i2 import metrics

# for type in TYPES:
#     data = load_dataset(f"han2024-{type}")
#     o2 = metrics["O2"]
#     try:
#         print(type, o2.ceiling(data).mean().item())
#     except ValueError as e:
#         print(e)

#     files = np.unique(data.stimulus_id.values)
#     for file in files:
#         if not((type, file) in ALL_PATHS):
#             print(f"{type}/{file} not found")