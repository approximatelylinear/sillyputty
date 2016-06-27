
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
from .util.odo_util import odo, odo_discover

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.path.abspath(os.getcwd())
PATH_DATA = os.getenv('PATH_MODEL_DATA') or os.path.join(THIS_DIR, 'data')


LOGGER = logging.getLogger(__name__)


def run(name, X, y=None, dshapes=None, model_config=None, uri_out=None, **kwargs):
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

    def __init__(self, model_config=None, namespace=None):
        #   Obtain model from the environment variable 'MODEL'
        self.model_obj = self._make_model(model_config, namespace=namespace)

    def _make_model(self, model_config, namespace=None):
        if namespace is None:
            namespace = globals()
        if model_config is None:
            path_model_config = os.getenv('MODEL_CONFIG')
            with open(path_model_config, 'rbU') as f_in:
                model_config = yaml.load(f_in)
        return model.make_model(model_config['model'], namespace=namespace)

    def _load_data(self, X, y=None, dshapes=None):
        """
        Load data using `odo`

        TBD: Adding dshapes causes error with odo when going from csv -> dataframe.
        """
        # if dshapes is None:
        #     dshapes = {}
        # dshape_X = dshapes.get('X') or odo_discover(X)
        # X = X if isinstance(X, np.ndarray) else odo.odo(X, pd.DataFrame, dshape=dshape_X).values
        X = X if isinstance(X, np.ndarray) else odo.odo(X, pd.DataFrame).values
        if y is not None:
            # dshape_y = dshapes.get('y') or odo_discover(y)
            # y = y if isinstance(y, np.ndarray) else odo.odo(y, pd.DataFrame, dshape=dshape_y).values
            y = y if isinstance(y, np.ndarray) else odo.odo(y, pd.DataFrame).values
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
        return res

    def _run(self, name, X, y=None, dshapes=None, uri_out=None, **kwargs):
        X, y = self._load_data(X, y, dshapes=dshapes)
        if y is not None:
            kwargs['y'] = y
        res = self._delegate(name, X, **kwargs)
        model.save_model(self.model_obj, os.path.join(PATH_DATA, 'model.dill'))
        if uri_out is not None:
            if res is not None:
                odo.odo(res, uri_out)
        else:
            return res

    def fit(self, X, y=None, dshapes=None, uri_out=None, **kwargs):
        return self._run('fit',
                    X,
                    y=y,
                    dshapes=dshapes,
                    uri_out=uri_out,
                    **kwargs
        )

    def fit_predict(self, X, y=None, dshapes=None, uri_out=None, **kwargs):
        return self._run('fit_predict',
                    X,
                    y=y,
                    dshapes=dshapes,
                    uri_out=uri_out,
                    **kwargs)

    def fit_transform(self, X, y=None, dshapes=None, uri_out=None, **kwargs):
        return self._run('fit_transform',
                    X,
                    y=y,
                    dshapes=dshapes,
                    uri_out=uri_out,
                    **kwargs)

    def predict(self, X, y=None, dshapes=None, uri_out=None, **kwargs):
        return self._run('predict',
                    X,
                    y=y,
                    dshapes=dshapes,
                    uri_out=uri_out,
                    **kwargs)

    def transform(self, X, y=None, dshapes=None, uri_out=None, **kwargs):
        return self._run('transform',
                    X,
                    y=y,
                    dshapes=dshapes,
                    uri_out=uri_out,
                    **kwargs)

    def score(self, X, y=None, dshapes=None, uri_out=None, **kwargs):
        return self._run('score',
                    X,
                    y=y,
                    dshapes=dshapes,
                    uri_out=uri_out,
                    **kwargs)

