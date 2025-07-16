import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import nnls 
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from .fdr import false_discovery_control
from ..evaluate.utils import split


NUM_PROCESSES = -1

class PositiveRegression:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        # add intercept
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        self.model = nnls(X, y)[0]
        return self
    
    def predict(self, X):
        # add intercept
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        return X @ self.model
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

def _cv(*data, fold_index, y_index=None):

    train_indices, test_indices = fold_index
    X, Y = data
    X_train = X[train_indices]
    X_test = X[test_indices]
    Y_train = Y[train_indices]
    Y_test = Y[test_indices]
    positive_regression = PositiveRegression()
    positive_regression.fit(X_train, Y_train)
    r2 = positive_regression.score(X_test, Y_test)

    return r2

# return [num_splits, ..., num_y_features]
def cv(X, Y, fold_index):
    from joblib import Parallel, delayed

    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)

    # normalize
    X = (X - X.mean(0)) / (X.std(0)+1e-10)
    Y = (Y - Y.mean(0)) / (Y.std(0)+1e-10)

    params = []
    for i in range(0, Y.shape[1]):
        params.append(((X, Y[:, i]), dict(y_index=i, fold_index=fold_index)))

    res = Parallel(n_jobs=NUM_PROCESSES)(
        delayed(_cv)(*args, **kwargs)
        for args, kwargs in params
    )

    res = np.array(res)
    return res


def analyze(X, Y, *X_factors, fold_index):
    r2_base = cv(X, Y, fold_index=fold_index)

    r2_remained = []
    for i, X_factor in enumerate(X_factors):
        r2 = cv(X_factor, Y, fold_index=fold_index)
        r2_remained.append(r2_base - r2)

    return r2_base, *r2_remained


def _resample_stats(data, confidence_level=0.95, axis=0):
    mean = np.mean(data, axis=axis)
    alpha = (1 - confidence_level)/2

    lower = np.percentile(data, alpha*100, axis=axis)
    upper = np.percentile(data, (1-alpha)*100, axis=axis)
    pval = np.mean(data < 0, axis=axis)

    return mean, lower, upper, pval


def resample_analyze(X, Y, *X_factors, n_resamples=1000, confidence_level=0.95, random_state=42):
    outs = []
    for random_seed in range(n_resamples):
        # print(f"Processing resample {n + 1}", end="\r")
        fold_index = generate_bootstrap_split(len(X), random_seed)
        out = analyze(X, Y, *X_factors, fold_index=fold_index)
        outs.append(out)

    outs = list(zip(*outs))
    outs = [np.stack(o, axis=0) for o in outs]
    stats = [_resample_stats(o, confidence_level, axis=0) for o in outs]

    return stats

def generate_bootstrap_split(num_samples, random_seed):
    random_state = np.random.RandomState(random_seed)
    indices = np.arange(num_samples)
    train_indices = random_state.choice(indices, size=num_samples, replace=True)
    test_indices = np.setdiff1d(indices, train_indices)
    return train_indices, test_indices


if __name__ == "__main__":

    n = 100
    p = 4
    X = abs(np.random.randn(n, p))
    X_f = X[:, 0]
    y = abs(np.random.randn(n)) * 0.2 + X[:, 0] * 0.5

    ret = resample_analyze(X, y, X_f)
    print(ret)
    breakpoint()