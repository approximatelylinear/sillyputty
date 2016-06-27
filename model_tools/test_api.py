

#   Stdlib
import os
import logging

#   3rd party
import yaml
import numpy as np

#   Custom current
from model_tools import model
from model_tools.api import Runner


LOGGER = logging.getLogger(__name__)

class TestCallable(object):
    def __init__(self, *args, **kwargs):
        super(TestCallable, self).__init__()

    def fit(self, X, y=None, **kwargs):
        print("Fitting with [ {} ] and [ {} ] ".format(X, y))

    def transform(self, X, y=None, **kwargs):
        print("Transforming with [ {} ] and [ {} ] ".format(X, y))
        return X

    def fit_transform(self, X):
        print("Fitting and Transforming with [ {} ] and [ {} ] ".format(X, y))
        return X


def test():
    model_config_yaml = """
    model:
        test:
            name: 'test'
            wrapper_class: 'model.Model'
            model_callable: 'TestCallable'
            parameters: {}
    """

    model_config = {
        'model': {
            'name': 'test',
            'wrapper_class': {
                'module': 'model_tools.model',
                'func': 'Model'
            },
            'model_callable': 'TestCallable',
            'parameters': {}
        }
    }


    runner = Runner(model_config, namespace=globals())
    X = np.random.randn(50, 2)
    y = np.ravel(np.random.randn(50, 1))
    runner.fit(X, y)
    runner.transform(X, y)
    runner.fit(X, y, dshapes={'X': 'var * 2 * float64', 'y': 'var * 1 * float64'})


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test()
