


#   Stdlib
import os
import logging
import inspect
import copy
try:
    import cPickle as pickle
except ImportError:
    import pickle
import pdb

#   3rd party

#   Custom current
# from .util.util import md5_hex, import_func
# from .util import odo_util
from model_tools.model import Model
from model_tools.util import odo_util
from model_tools.util.odo_util import odo, odo_convert, odo_resource, odo_append

LOGGER = logging.getLogger(__name__)

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.path.abspath(os.getcwd())

PATH_DATA = os.getenv('PATH_MODEL_DATA') or os.path.join(THIS_DIR, 'data')


"""
Save components
    - Truncate model
        - Create model properties object
    - Individually specified uris
        - specified in the config
            - inspect the config
        - Use odo to save items to their uris

Load components
    - Individually specified uris
        - retrieve uris
            - specified in config
                - inspect the config
        - load uris
            - odo
            - create model objects
                - each saved item should be a ModelProperties instance
                    - convention:
                        - <filename>.modelproperties.<type>
                - expand model properties


Model notes
- Save submodels recursively
- Reconstruct submodels recursively
- List of all submodels
- Each submodel is it's own property definition
    - Root object is the model object itself
        - Root
            - name
            - <address>
                - Stored as a hidden file of the same name
        - Submodels
            - Dictionary of properties with addresses of model objects
                {
                    <name> : <address>,
                    ...,
                    <name>: <address>
                }

- Mechanism
    - Model.save()
        - Classmethod
        - Load model properties
        - Return a new model instance
    - Model.load()
        - Classmethod
        - Create model properties


Tests
- save
    - Create model properties
        - Single
        - Nested
            - 1
            - 10
            - 100
    - Write to file
        - local
            - pickle
                - file pickle to model_properties
        - s3
            - pickle
- load
    - Read from file
        - local
            - pickle
        - s3
            - pickle
    - Restore model properties

"""


class EstimatorPickler(object):
    _save_attrs = []

    def __init__(self, save_attrs=None):
        self.save_attrs = save_attrs or self._save_attrs

    def save(self, obj, path):
        if not os.path.exists(path):
            os.makedirs(path)
        paths = self._save_state(obj, path)
        obj_copy = clone(obj)
        for attr, path_attr in paths.iteritems():
            setattr(obj_copy, attr, path_attr)
        path_instance = os.path.join(path, 'instance.pkl')
        with open(path_instance, 'wb') as f_out:
            pickle.dump(obj_copy, f_out)

    def _save_state(self, obj, path):
        paths = {}
        for attr in self.save_attrs:
            path_attr = os.path.join(path, u'.{}'.format(attr))
            with open(path_attr, 'wb') as f_out:
                val = getattr(obj, attr, None)
                if val is not None:
                    pickle.dump(val, f_out)
                    paths[attr] = path_attr
        return paths

    def load(self, path):
        path_instance = os.path.join(path, 'instance.pkl')
        with open(path_instance, 'rb') as f_in:
            obj = pickle.load(f_in)
        self._load_state(obj, path)
        return obj

    def _load_state(self, obj, path):
        for attr in self.save_attrs:
            path_attr = getattr(obj, attr)
            if os.path.exists(path_attr):
                with open(path_attr, 'rb') as f_in:
                    val = pickle.load(f_in)
                    setattr(obj, attr, val)
            else:
                print("Could not find file for attribute {} at {}".format(attr, path_attr))


"""
import os
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import make_low_rank_matrix

X = make_low_rank_matrix()
model_svd = TruncatedSVD()
model_svd.fit(X)

SVDEstimatorPickler = EstimatorPickler(save_attrs=['components_', 'explained_variance_', 'explained_variance_ratio_'])
SVDEstimatorPickler.save(model_svd, path='test_svd_pkl')
svd2 = SVDEstimatorPickler.load(path='test_svd_pkl')

"""


def test_persist():
    model_a = Model(
        config={
            'name': 'a',
            'model_callable': 'TestCallable'
        })
    model_b = Model(
        config={
            'name': 'c',
            'model_callable': 'TestCallable',
            'model': [
                {
                    'name': 'a',
                    'model_callable': 'TestCallable',
                },
                {
                    'name': 'b',
                    'model_callable': 'TestCallable',
                },
            ]
        }
    )
    model_props_maker = ModelProperties()
    model_props_a = model_props_maker.compress(model_a)
    print model_props_a
    model_props_b = model_props_maker.compress(model_b)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_persist()
