#   Stdlib
import os

#   3rd party
import yaml
import numpy as np
import pandas as pd
import odo

#   Custom current
from . import model


def fit(X, y=None, model_config=None, path_out=None, **kwargs):
    return _run('fit',
                X,
                y=y,
                model_config=model_config,
                path_out=path_out,
                **kwargs)


def fit_predict(X, y=None, model_config=None, path_out=None, **kwargs):
    return _run('fit_predict',
                X,
                y=y,
                model_config=model_config,
                path_out=path_out,
                **kwargs)


def fit_transform(X, y=None, model_config=None, path_out=None, **kwargs):
    return _run('fit_transform',
                X,
                y=y,
                model_config=model_config,
                path_out=path_out,
                **kwargs)


def predict(X, y=None, model_config=None, path_out=None, **kwargs):
    return _run('predict',
                X,
                y=y,
                model_config=model_config,
                path_out=path_out,
                **kwargs)


def transform(X, y=None, model_config=None, path_out=None, **kwargs):
    return _run('transform',
                X,
                y=y,
                model_config=model_config,
                path_out=path_out,
                **kwargs)


def score(X, y=None, model_config=None, path_out=None, **kwargs):
    return _run('score',
                X,
                y=y,
                model_config=model_config,
                path_out=path_out,
                **kwargs)


def _run(name, X, y=None, model_config=None, path_out=None, **kwargs):
    def __make_model():
        if model_config is None:
            path_model_config = os.getenv('MODEL_CONFIG')
            with open(path_model_config, 'rbU') as f_in:
                model_config = yaml.load(f_in)
        return model.make_model(model_config['model'], globals_=globals())

    def __delegate(name, *args, **kwargs):
        #   Obtain model from the environment variable 'MODEL'
        model_obj = ml_main.make_model()
        func = getattr(model_obj, name)
        res = func(*args, **kwargs)
        model_obj.save()
        return res

    def __load_data(X, y=None):
        X = X if isinstance(X, np.ndarray) else odo.odo(
            X, pd.DataFrame).fillna('').values()
        if y is not None:
            y = y if isinstance(y, np.ndarray) else odo.odo(
                y, pd.DataFrame).fillna('').values()
        return X, y

    X, y = __load_data(X, y)
    if y is not None:
        kwargs['y'] = y
    res = __delegate(name, X, **kwargs)
    if path_out is not None:
        odo.odo(res, path_out)
    else:
        return res

    return __func
