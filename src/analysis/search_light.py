from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_data
import numpy as np
from scipy.sparse import csr_matrix
import os


import pickle

def generate_adj(subject='fsaverage5'):

    path = f'src/analysis/cache/{subject}'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    fsaverage = fetch_surf_fsaverage(subject)
    l_conn = load_surf_data(fsaverage.infl_left)[1]
    r_conn = load_surf_data(fsaverage.infl_right)[1] + 10242
    full_conn = np.concatenate([l_conn, r_conn], axis=0)

    # Data [N, P]
    # sparse_conn [P, P] -> Data @ sparse_conn = Data'

    # sparse matrix
    sparse_conn = csr_matrix((20484, 20484))
    for _, conn in enumerate(full_conn):
        print(_, end="\r")
        i, j, v = conn
        sparse_conn[i, j] = 1
        sparse_conn[j, i] = 1
        sparse_conn[i, i] = 1
        sparse_conn[j, j] = 1
        sparse_conn[i, v] = 1   
        sparse_conn[j, v] = 1
        sparse_conn[v, i] = 1
        sparse_conn[v, j] = 1
        sparse_conn[v, v] = 1

    with open(path, 'wb') as f:
        pickle.dump(sparse_conn, f)

    return sparse_conn

def generate_power_adj(subject='fsaverage5', power=5):
    adj = generate_adj(subject)
    for i in range(power):
        adj = adj @ adj
        adj = (adj>0).astype(int)

    return adj


# m = generate_adj()

# breakpoint()