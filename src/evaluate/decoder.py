import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from .utils import pearsonr, take_layers, NUM_PROCESSES, pack_multi_np_rets, copy_random_state, ScaledLogisticRegression


def _filter_inf_nan(data):
    ret = []
    for X, y in data:
        X = X.astype(np.float32)
        try:
            y = y.astype(np.float32)
        except:
            pass
        inf_rows = np.any(np.isinf(X), axis=1)
        nan_rows = np.any(np.isnan(X), axis=1)
        mask = ~(inf_rows | nan_rows)
        ret.append((X[mask], y[mask]))
    return ret

def _normalize_x(train, *tests):
    X_train, y_train = train
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    train = X_train, y_train
    tests = [(X_scaler.transform(X), y) for X, y in tests]
    return train, tests

def _score_regression(regressor):
    def score(X, Y):
        Y_ = regressor.predict(X)
        return pearsonr(Y, Y_)
    return score

def _preprocess_data(train, *tests):
    train = _filter_inf_nan([train])[0]
    tests = _filter_inf_nan(tests)
    train, tests = _normalize_x(train, *tests)
    train = _filter_inf_nan([train])[0]
    tests = _filter_inf_nan(tests)
    return train, tests

def ridgecv(train, *tests, alphas=np.logspace(-4, 4, 9)):
    train, tests = _preprocess_data(train, *tests)
    clf = RidgeCV(alphas=alphas)
    clf.fit(*train)
    test_scores = [_score_regression(clf)(*test) for test in tests]
    return test_scores

def pls(train, *tests, n_components=25):
    train, tests = _preprocess_data(train, *tests)
    clf = PLSRegression(n_components=n_components)
    clf.fit(*train)
    test_scores = [_score_regression(clf)(*test) for test in tests]
    return test_scores

def logistic_regression(train, *tests, C=1.0):
    train, tests = _preprocess_data(train, *tests)
    train = _flatten_y(*train)
    tests = [_flatten_y(*test) for test in tests]
    clf = ScaledLogisticRegression(C=C)
    clf.fit(*train)
    test_scores = [clf.score(*test) for test in tests]
    return test_scores

def _flatten_y(X, Y):
    if Y.shape[1] == 1 and Y.ndim == 2:
        Y = Y.ravel()
    return X, Y

decoders = {
    "ridgecv": ridgecv,
    "pls": pls,
    "logistic_regression": logistic_regression,
}