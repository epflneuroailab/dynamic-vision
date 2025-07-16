from nilearn.glm.first_level import spm_hrf
import numpy as np
from scipy.signal import convolve
from brainio.assemblies import NeuroidAssembly
import torch
from torch.nn.functional import conv1d

def convolve_hrf(response, tr):
    hrf = spm_hrf(tr, oversampling=1)
    padding_left = np.zeros(len(hrf) - 1)
    response = np.concatenate([padding_left, response])
    ret = np.convolve(response, hrf, mode="valid")
    return ret

def convolve_hrf_batch(response, tr):
    N = response.shape[0]
    hrf = spm_hrf(tr, oversampling=1)
    padding_left = np.zeros((N, len(hrf)-1))
    response = np.concatenate([padding_left, response], axis=1)

    response = torch.tensor(response).float()
    hrf = torch.tensor(hrf).float()
    hrf = torch.flip(hrf, [0])
    ret = conv1d(response[:, None, :], hrf[None, None, :])
    ret = ret.numpy()[:, 0, :]
    return ret


def convolve_hrf_assembly(asm, tr):
    asm = asm.transpose("stimulus_path", "neuroid", "time_bin")
    response = asm.values
    P, N, T = response.shape
    response = response.reshape(P * N, T)
    ret = convolve_hrf_batch(response, tr)
    ret = ret.reshape(P, N, T)
    asm[:] = ret
    return asm


def test():
    response = np.zeros(50)
    response[-2] = 1.0
    tr = 2.0
    ret = convolve_hrf(response, tr)
    print(ret.shape)
    print(ret)

    response = np.zeros((2, 50))
    response[:, -2] = 1.0
    tr = 2.0
    ret = convolve_hrf_batch(response, tr)
    print(ret.shape)
    print(ret)

    asm = NeuroidAssembly(
        np.ones((100, 40000, 200)),
        coords={
            "presentation": ("presentation", np.arange(100)),
            "neuroid": ("neuroid", np.arange(40000)),
            "time_bin": ("time_bin", np.arange(200)),
        }
    )

    asm = convolve_hrf_assembly(asm, tr)

    print(asm.values)