
#   Stdlib
import os
import logging
import pdb

#   3rd party
import yaml
import numpy as np
import pandas as pd

#   Custom current
from . import model
from .util.odo_util import odo, discover


LOGGER = logging.getLogger(__name__)


def run(name, X, y=None, dshapes=None, model_config=None, path_out=None, **kwargs):
    pass



class Runner(object):
    """
    runner = Runner(model_config)
    runner = Runner('my_model.yaml')
    runner.fit(np.random.randn(50, 2), np.random.randn(50, 1))
    runner.fit(np.random.randn(50, 2), np.random.randn(50, 1), dshapes={'X': 'var * 2 * float64', 'y': 'var * 1 * float64'})
    pd.DataFrame(np.random.randn(50, 2)).to_csv('test_X.csv', index=False)
    pd.DataFrame(np.random.randn(50, 1)).to_csv('test_y.csv', index=False)
    runner.fit('test_X.csv', 'test_y.csv', dshapes={'X': 'var * 2 * float64', 'y': 'var * 1 * float64'})
    """

    def __init__(self, model_config=None, globals_=None):
        #   Obtain model from the environment variable 'MODEL'
        self.model_obj = self._make_model(model_config, globals_=globals_)

    def _make_model(self, model_config, globals_=None):
        if globals_ is None:
            globals_ = globals()
        if model_config is None:
            path_model_config = os.getenv('MODEL_CONFIG')
            with open(path_model_config, 'rbU') as f_in:
                model_config = yaml.load(f_in)
        return model.make_model(model_config['model'], globals_=globals_)

    def _load_data(self, X, y=None, dshapes=None):
        """
        Load data using `odo`
        """
        if dshapes is None:
            dshapes = {}
        dshape_X = dshapes.get('X') or discover(X)
        X = X if isinstance(X, np.ndarray) else odo.odo(X, pd.DataFrame, dshape=dshape_X).values
        if y is not None:
            dshape_y = dshapes.get('y') or discover(y)
            y = y if isinstance(y, np.ndarray) else odo.odo(y, pd.DataFrame, dshape=dshape_y).values
            #   Squeeze y to a 1d array, per standard conventions.
            if len(y.shape) > 1 and y.shape[1] == 1:
                y = np.ravel(y)
        return X, y

    def _delegate(self, name, *args, **kwargs):
        """
        Run a function that is a member of `self.model_obj`
        """
        func = getattr(self.model_obj, name)
        res = func(*args, **kwargs)
        self.model_obj.save()
        return res

    def _run(self, name, X, y=None, dshapes=None, path_out=None, **kwargs):

        pdb.set_trace()

        X, y = self._load_data(X, y, dshapes=dshapes)
        if y is not None:
            kwargs['y'] = y
        res = self._delegate(name, X, **kwargs)
        if path_out is not None:
            odo.odo(res, path_out)
        else:
            return res

    def fit(self, X, y=None, dshapes=None, path_out=None, **kwargs):
        return self._run('fit',
                    X,
                    y=y,
                    dshapes=dshapes,
                    path_out=path_out,
                    **kwargs
        )

    def fit_predict(self, X, y=None, dshapes=None, path_out=None, **kwargs):
        return self._run('fit_predict',
                    X,
                    y=y,
                    dshapes=dshapes,
                    path_out=path_out,
                    **kwargs)

    def fit_transform(self, X, y=None, dshapes=None, path_out=None, **kwargs):
        return self._run('fit_transform',
                    X,
                    y=y,
                    dshapes=dshapes,
                    path_out=path_out,
                    **kwargs)

    def predict(self, X, y=None, dshapes=None, path_out=None, **kwargs):
        return self._run('predict',
                    X,
                    y=y,
                    dshapes=dshapes,
                    path_out=path_out,
                    **kwargs)

    def transform(self, X, y=None, dshapes=None, path_out=None, **kwargs):
        return self._run('transform',
                    X,
                    y=y,
                    dshapes=dshapes,
                    path_out=path_out,
                    **kwargs)

    def score(self, X, y=None, dshapes=None, path_out=None, **kwargs):
        return self._run('score',
                    X,
                    y=y,
                    dshapes=dshapes,
                    path_out=path_out,
                    **kwargs)

