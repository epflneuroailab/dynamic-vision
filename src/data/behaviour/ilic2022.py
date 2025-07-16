import os
import pandas as pd
import random
import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.stimuli import StimulusSet


UCF101_DATA_DIR = "/home/ytang/workspace/data/ucf101/UCF-101"
AFD101_DATA_DIR = "/home/ytang/workspace/data/AFD/afd101"
RESPONSE_DATA_DIR = "/home/ytang/workspace/data/ilic2022/response"
MAX_VIDEO_PER_CONDITION = 4  # same as in the paper


def get_user_csv(root_dir, username):
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            if name == f'{username}.csv':
                print(f'{name}: == {os.path.join(path, name)}')
                csv_file = os.path.join(path, name)
                return pd.read_csv(csv_file)
    raise ValueError("couldnt find user")


def collect_all_user_data(root_dir, target_dataset="afd5", ignore_index=True):
    all_csv_files = []
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            if name.endswith('.csv'):
                csv_file = os.path.join(path, name)
                print(csv_file)
                all_csv_files.append(csv_file)
    df = pd.concat(map(pd.read_csv, all_csv_files), ignore_index=ignore_index)

    df["Prediction"] = df["Prediction"].apply(lambda s: s if "<---" not in s else np.nan)
    df = df.drop("Unnamed: 0", axis=1)
    df = df.dropna(axis=0, how='any')
    df["Modality"] = df["Modality"].apply(lambda s: s if s=='ucf5' else 'afd5')

    def change_path(s):
        s = s.replace("data/TRIAL", "data/")
        s = s.replace("data/ucf5af", "afd")
        s = s.replace("data/ucf5", "ucf")  # don't change order!
        return s
    df["VideoPath"] = df["VideoPath"].apply(change_path)

    df = df[df.Modality == target_dataset]
    return df


def presentation_matrix(df):
    classes = df.GroundTruth.unique()
    matrix = pd.DataFrame(index=classes, columns=classes)
    matrix = matrix.fillna(0)
    for i in range(len(df)):
        matrix.loc[df.GroundTruth.values[i], df.Prediction.values[i]] += 1
    return matrix


# df = collect_all_user_data("./response")
# ucf_matrix = presentation_matrix(df[df.Modality=='ucf5'])
# afd_matrix = presentation_matrix(df[df.Modality=='afd5'])


def load_dataset(identifier):
    if identifier == "ilic2022-ucf5":
        target_dataset = "ucf5"
        data_root = UCF101_DATA_DIR
    elif identifier == "ilic2022-afd5":
        target_dataset = "afd5"
        data_root = AFD101_DATA_DIR

    df = collect_all_user_data(RESPONSE_DATA_DIR, target_dataset)

    def change_path(s):
        s = s.replace("ucf", data_root)
        s = s.replace("afd", data_root)
        return s

    stimulus_ids = df["VideoPath"].values
    subjects = df["UserID"].values
    truths = df["GroundTruth"].values

    conditions = [target_dataset]*len(stimulus_ids)

    assembly = BehavioralAssembly(
        df["Prediction"].values,
        dims=['presentation'],
        coords={
            "stimulus_id": ("presentation", stimulus_ids),
            "condition": ("presentation", conditions),
            "subject": ("presentation", subjects),
            "truth": ("presentation", truths),
        }
    )

    # remove repeated stimuli
    df = df.drop_duplicates(subset=["VideoPath"])
    stimulus_paths = df["VideoPath"].apply(change_path).values
    stimulus_ids = df["VideoPath"].values
    subjects = df["UserID"].values
    truths = df["GroundTruth"].values

    stimulus_set = {}
    stimulus_set["stimulus_id"] = stimulus_ids
    stimulus_set['truth'] = truths
    stimulus_set['subject'] = subjects
    stimulus_set = StimulusSet(stimulus_set)
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}
    stimulus_set.identifier = identifier

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly


def load_stimulus_set(identifier):
    assert identifier in ["ilic2022-afd5-fitting_stimuli", "ilic2022-ucf5-fitting_stimuli"]
    testset = load_dataset(identifier[:-len('-fitting_stimuli')]).stimulus_set
    classes = np.unique(testset["truth"].values)
    avoid_paths = testset.stimulus_paths.values()

    if identifier == "ilic2022-afd5-fitting_stimuli":
        video_dir = AFD101_DATA_DIR
    elif identifier == "ilic2022-ucf5-fitting_stimuli":
        video_dir = UCF101_DATA_DIR
    data = []
    paths = {}
    for cls in classes:
        action_dir = os.path.join(video_dir, cls)
        videos = os.listdir(action_dir)
        count = 0
        for v in videos:
            if count >= MAX_VIDEO_PER_CONDITION:
                break
            if os.path.join(action_dir, v) in avoid_paths:
                print(f"avoiding {os.path.join(action_dir, v)}")
                continue
            count += 1
            data.append({"stimulus_id": v, "truth": cls})
            paths[v] = os.path.join(action_dir, v)
    
    stimulus_set = StimulusSet(data)
    stimulus_set.stimulus_paths = paths
    stimulus_set.identifier = identifier
    return stimulus_set