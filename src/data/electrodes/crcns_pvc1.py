import os
import numpy as np
from scipy.io import loadmat
import re
import h5py
from brainio.assemblies import NeuroidAssembly
from brainio.stimuli import StimulusSet


DATA_DIR = "/home/ytang/workspace/data/yourhead/crcns-pvc1/neurodata/ac1"
MOVIE_PATH  = "/home/ytang/workspace/data/yourhead/crcns-pvc1/movies.h5"
VIDEO_PATH = "/home/ytang/workspace/data/yourhead/crcns-pvc1/videos"
os.makedirs(VIDEO_PATH, exist_ok=True)
TARGET_TIME_BIN = 200  # ms
FPS = 30


def load_dataset(identifier="crcns-pvc1"):
    frs, all_vids, all_elecs = _get_frs()
    all_movies = _get_movies()
    stimulus_ids = []
    stimulus_paths = []
    movies = []
    segments = []
    for i, (mov, seg) in enumerate(all_vids):
        print(f"{i+1}/{len(all_vids)}")
        frames = all_movies[(mov, seg)]
        frames = frames[:int(30000/1000*FPS)].transpose(0, 2, 3, 1)
        vid_path = os.path.join(VIDEO_PATH, f"{mov}_{seg}.avi")
        write_video(vid_path, frames, FPS)

        stimulus_ids.append(f"{mov}_{seg}")
        stimulus_paths.append(vid_path)
        movies.append(mov)
        segments.append(seg)

    frs # presentation x repetition x neuroid x time
    P, R, N, T = frs.shape

    assembly = NeuroidAssembly(
        frs,
        dims=('stimulus_id', 'repetition', 'neuroid', 'time_bin'),
        coords={
            'neuroid_id': ("neuroid", np.arange(N)),
            'electrode_id': ("neuroid", all_elecs),
            'time_bin_start': ("time_bin", (np.arange(T))*TARGET_TIME_BIN),
            'time_bin_end': ("time_bin", (1+np.arange(T))*TARGET_TIME_BIN),
            'stimulus_id': ("stimulus_id", stimulus_ids),
            'repetition': ("repetition", np.arange(R)),
        },
    )

    assembly = assembly.assign_coords(movie=('stimulus_id', movies))
    assembly = assembly.assign_coords(segment=('stimulus_id', segments))
    assembly = assembly.stack(presentation=("stimulus_id", "repetition"))
    assembly = assembly.__class__(assembly)

    # make stimulus set
    stimulus_set = StimulusSet([{'stimulus_path': path, "stimulus_id": i} 
                                for i, path in zip(stimulus_ids, stimulus_paths)])
    stimulus_set.identifier = f'crcns-pvc1'
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly


def _get_movies():
    mov_pattern = re.compile(r"movie(\d{3})_(\d{3})")
    with h5py.File(MOVIE_PATH, 'r') as f:
        movies = {}
        for k in f.keys():
            mov, seg = mov_pattern.match(k).groups()
            mov = int(mov)
            seg = int(seg)
            movies[(mov, seg)] = f[k][()]
    return movies


def _get_fr(spiketimes, bin_time=TARGET_TIME_BIN, max=30000):
    spiketimes = spiketimes * 1000
    spiketimes = spiketimes[spiketimes < max]
    bins = np.bincount((spiketimes // bin_time).astype(int), minlength=max // bin_time)
    return bins

class PVC1NeuroData(dict): 
    def __init__(self, pvc1_data_path):
        d = loadmat(pvc1_data_path)['pepANA'][0,0]
        electrodes = d[-1][0].tolist()
        trials = d[6][0]
        data = {}
        print(os.path.basename(pvc1_data_path), len(trials))
        for t in trials:
            t = t[0][0]
            condition = tuple(t[0][0].tolist()[:2])
            for eid in range(len(electrodes)):
                spiketimes = t[4][0,0][0,0][4][0][eid][0][0][0]
                data[(*condition, electrodes[eid])] = spiketimes
        super().__init__(data)


def _get_frs():

    all_data = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".mat"):
            path = os.path.join(DATA_DIR, filename)
            data = PVC1NeuroData(path)
            all_data.append(data)

    all_keys = set()
    for d in all_data:
        for k in d.keys():
            all_keys.add(k)
    all_vids = set([(mov, seg) for mov, seg, elec in all_keys])
    all_elecs = set([elec for mov, seg, elec in all_keys])

    presence_matrix = np.zeros((len(all_vids), len(all_elecs)))

    for d in all_data:
        for k in d.keys():
            mov, seg, elec = k
            vid_idx = list(all_vids).index((mov, seg))
            elec_idx = list(all_elecs).index(elec)
            presence_matrix[vid_idx, elec_idx] += 1


    rows_to_discard = (presence_matrix<=1).sum(axis=1) > presence_matrix.shape[1]//2
    all_vids = [vid for i, vid in enumerate(all_vids) if not rows_to_discard[i]]
    presence_matrix = presence_matrix[~rows_to_discard]

    cols_to_discard = (presence_matrix<=1).sum(axis=0) > presence_matrix.shape[0]//2
    all_elecs = [elec for i, elec in enumerate(all_elecs) if not cols_to_discard[i]]
    presence_matrix = presence_matrix[:,~cols_to_discard]

    # take 2 repetition

    T = _get_fr(all_data[0][list(all_data[0].keys())[0]]).shape[0]
    frs = np.zeros((len(all_vids), 2, len(all_elecs), T))
    stimulus_meta = [(mov, seg) for mov, seg in all_vids]
    count = {}
    for d in all_data:
        for k in d.keys():
            mov, seg, elec = k
            if (mov, seg) not in all_vids or elec not in all_elecs:
                continue
            count[k] = count.get(k, 0) + 1
            if count[k] > 2:
                continue
            frs[all_vids.index((mov, seg)), count[k]-1, all_elecs.index(elec)] = _get_fr(d[k])

    P = len(all_vids)
    N = len(all_elecs)
    corrs = []
    corrs_at = []
    for n in range(N):
        corrs.append(np.corrcoef(frs[:,0,n].flatten(), frs[:,1,n].flatten())[0,1])
        corrs_at.append(np.corrcoef(frs[:,0,n].sum(-1).flatten(), frs[:,1,n].sum(-1).flatten())[0,1])
    corrs = np.array(corrs)
    corrs_at = np.array(corrs_at)

    # filter correlation
    valid = corrs > 0.065
    frs = frs[:, :, valid]
    all_elecs = [elec for i, elec in enumerate(all_elecs) if valid[i]]

    return frs, all_vids, all_elecs


# find the maximal complete subset matrix, the size is defined by #videos x #elecs x #repe

def compute_size(rows, cols):
    if rows == []:
        return 0
    matrix = presence_matrix[rows][:,cols]
    if (matrix < 1).any() or len(cols) < 5:
        return 0
    return len(rows) * len(cols) * matrix.min()

def memorize(func):
    cache = {}
    def wrapper(*args):
        ll = [list(a) for a in args]
        for l in ll: l.sort()
        wargs = tuple([tuple(l) for l in ll])
        if wargs not in cache:
            cache[wargs] = func(*args)
        return cache[wargs]
    return wrapper

@memorize
def find_largest_subset(rows, cols):
    if rows == [] or cols == []:
        return [], []

    this_size = compute_size(rows, cols)

    results = [(this_size, (rows, cols))]

    # case 1:
    for i in range(len(rows)):
        nrows = rows.copy()
        del nrows[i]
        case_1_rows, case_1_cols = find_largest_subset(nrows, cols)
        case_1size = compute_size(case_1_rows, case_1_cols)
        results.append((case_1size, (case_1_rows, case_1_cols)))

    # case 2:
    for i in range(len(cols)):
        ncols = cols.copy()
        del ncols[i]
        case_2_rows, case_2_cols = find_largest_subset(rows, ncols)
        case_2size = compute_size(case_2_rows, case_2_cols)
        results.append((case_2size, (case_2_rows, case_2_cols)))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[0][1]


def write_video(path, video, fps):
    # write to mp4; video: np.array
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (video.shape[2], video.shape[1]))
    for frame in video:
        out.write(frame[..., ::-1])
    out.release()
