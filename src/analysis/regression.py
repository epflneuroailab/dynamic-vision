import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import nnls 
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression

from .fdr import false_discovery_control
from ..evaluate.utils import split


NUM_PROCESSES = 1


def positive_regression(X, y):
    # add intercept
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    w_, _ = nnls(X, y)
    y_pred = X @ w_
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (X.shape[0] - 1) / (X.shape[0] - X.shape[1] - 1)
    
    w = w_[1:]

    return w, adj_r2

def _cv(*data, y_index=None, random_seed=42):

    random_state = np.random.RandomState(random_seed)
    N = data[0].shape[0]
    indices = random_state.choice(np.arange(N), N, replace=True)
    data = [d[indices] for d in data]
    stat = positive_regression(*data)

    # if y_index is not None:
    #     print(f"Processing target {y_index + 1}", end="\r")

    return stat

# return [num_splits, ..., num_y_features]
def cv(X, Y, random_seed):
    from joblib import Parallel, delayed

    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)

    # normalize
    X = (X - X.mean(0)) / (X.std(0)+1e-10)
    Y = (Y - Y.mean(0)) / (Y.std(0)+1e-10)

    params = []
    for i in range(0, Y.shape[1]):
        params.append(((X, Y[:, i]), dict(y_index=i, random_seed=random_seed)))

    res = Parallel(n_jobs=NUM_PROCESSES)(
        delayed(_cv)(*args, **kwargs)
        for args, kwargs in params
    )

    res = list(zip(*res))
    res = [np.stack(r, -1) for r in res]
    return res


def analyze(X, Y, *X_factors, random_seed=42):
    w_base, r2_base = cv(X, Y, random_seed=random_seed)

    r2_remained = []
    for i, X_factor in enumerate(X_factors):
        w, r2 = cv(X_factor, Y, random_seed=random_seed)
        r2_remained.append(r2_base - r2)

    return w_base, r2_base, *r2_remained


def _resample_stats(data, confidence_level=0.95, axis=0):
    mean = np.mean(data, axis=axis)
    alpha = (1 - confidence_level)/2

    lower = np.percentile(data, alpha*100, axis=axis)
    upper = np.percentile(data, (1-alpha)*100, axis=axis)

    return mean, lower, upper


def resample_analyze(X, Y, *X_factors, n_resamples=1000, confidence_level=0.95):
    outs = []
    for n in range(n_resamples):
        print(f"Processing resample {n + 1}", end="\r")
        out = analyze(X, Y, *X_factors, random_seed=n)
        outs.append(out)

    outs = list(zip(*outs))
    outs = [np.stack(o, axis=0) for o in outs]
    stats = [_resample_stats(o, confidence_level, axis=0) for o in outs]

    return stats


if __name__ == "__main__":

    n = 100
    p = 4
    X = abs(np.random.randn(n, p))
    X_f = X[:, 0]
    y = abs(np.random.randn(n)) * 0.2 + X[:, 0] * 0.5

    ret = resample_analyze(X, y, X_f)
    print(ret)
    breakpoint()