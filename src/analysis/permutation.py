import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import nnls 
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression

from .fdr import false_discovery_control
from ..evaluate.utils import split


NUM_PROCESSES = -1


def positive_regression(X, y):
    # add intercept
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    w_, _ = nnls(X, y)
    y_pred = X @ w_
    r2 = r2_score(y, y_pred)
    
    w = w_[1:]

    return w, r2

def _cv(*data, y_index=None):
    stat = positive_regression(*data)

    # if y_index is not None:
    #     print(f"Processing target {y_index + 1}", end="\r")

    return stat

# return [num_splits, ..., num_y_features]
def cv(X, Y):
    from joblib import Parallel, delayed

    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)

    # normalize
    X = (X - X.mean(0)) / (X.std(0)+1e-10)
    Y = (Y - Y.mean(0)) / (Y.std(0)+1e-10)

    params = []
    for i in range(0, Y.shape[1]):
        params.append(((X, Y[:, i]), dict(y_index=i)))

    res = Parallel(n_jobs=NUM_PROCESSES)(
        delayed(_cv)(*args, **kwargs)
        for args, kwargs in params
    )

    res = list(zip(*res))
    res = [np.stack(r, -1) for r in res]
    return res

def adjust_r2(r2, X):
    n, p = X.shape
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return r2_adj

def analyze(X, Y, *X_factors, random_seed=42):
    r2_diffs = []

    for X_factor in X_factors:
        X_combined = np.hstack([X, X_factor])
        _, r2_combined = cv(X_combined, Y)
        _, r2_factor = cv(X_factor, Y)
        r2_combined_adj = adjust_r2(r2_combined, X_combined)
        r2_factor_adj = adjust_r2(r2_factor, X_factor)
        r2_diff = r2_combined_adj - r2_factor_adj  # subtracting extra explained variance of the factor
        r2_diffs.append(r2_diff)

    _, r2_self = cv(X, Y)
    r2_self_adj = adjust_r2(r2_self, X)

    return r2_self_adj, *r2_diffs

def _permutation_stats(true_sample, perm_sample, confidence_level=0.95, axis=0):
    mean = true_sample
    pval = np.mean(perm_sample > np.expand_dims(true_sample, axis), axis=axis)
    pval = false_discovery_control(pval)
    return mean, pval

def permutation_analyze(X, Y, *X_factors, n_permutations=1000, confidence_level=0.95):
    true_out = analyze(X, Y, *X_factors)

    sample_outs = []
    for n in range(n_permutations):
        print(f"Processing permutation {n + 1}", end="\r")
        # permute X to study the effect of it
        X_perm = X[np.random.permutation(X.shape[0])]
        out = analyze(X_perm, Y, *X_factors, random_seed=n)
        sample_outs.append(out)

    sample_outs = list(zip(*sample_outs))
    sample_outs = [np.stack(o, axis=0) for o in sample_outs]

    stats = [_permutation_stats(to, so, confidence_level, axis=0) for to, so in zip(true_out, sample_outs)]

    return stats


if __name__ == "__main__":

    n = 100
    p = 4
    X = abs(np.random.randn(n, p))
    X_f = X[:, 1:2]
    y = abs(np.random.randn(n)) * 0.2 + X[:, 0] * 0.5

    ret = permutation_analyze(X, y, X_f)
    print(ret)
    breakpoint()