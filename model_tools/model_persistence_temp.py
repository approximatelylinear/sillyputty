


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
"""


# @odo_convert.register(PICKLE, Model)
def model_to_pickle(model_obj):
    # model_properties = odo_convert(ModelProperties, model_obj)
    pass


def save(path, obj):
    print("Saving {} to {}".format(path, repr(obj)))


def save_model_properties(model_properties):
    #   Keep track of the model_properties object
    root_name = model_properties['root_name']
    root_path = model_properties['props'][root_name].get('path_model')
    if root_path is None:
        pass
        #   Where should the root obj be saved?
    basic_properties = {
        'root_name': root_name,
        'props': {} }

    #   Save individual model properties
    for name, props in model_properties['props']:
        path_model = props.get('path_model') or '{}.model.pkl'.format(name)
        props = {
            'root_name': root_name,
            'props': basic_properties['props'][name]
        }
        # odo_convert(path_model, props)
        save(path_model, props)
        #   Add info to the basic properties, so it can be reconstructed
        basic_properties['props'][name]['path_model'] = path_model

    #   Save the model_properties object
    #   TBD: Where should this be saved? This has the most important info.
    #       - To the data path, I guess


# Model notes
# - Save submodels recursively
# - Reconstruct submodels recursively
# - List of all submodels
# - Each submodel is it's own property definition
#     - Root object is the model object itself
#         - Root
#             - name
#             - <address>
#                 - Stored as a hidden file of the same name
#         - Submodels
#             - Dictionary of properties with addresses of model objects
#                 {
#                     <name> : <address>,
#                     ...,
#                     <name>: <address>
#                 }

# - Mechanism
#     - Model.save()
#         - Classmethod
#         - Load model properties
#         - Return a new model instance
#     - Model.load()
#         - Classmethod
#         - Create model properties


# Tests
# - save
#     - Create model properties
#         - Single
#         - Nested
#             - 1
#             - 10
#             - 100
#     - Write to file
#         - local
#             - pickle
#                 - file pickle to model_properties
#         - s3
#             - pickle
# - load
#     - Read from file
#         - local
#             - pickle
#         - s3
#             - pickle
#     - Restore model properties



class ModelProperties(object):
    """
    Truncated form of model permitting easy reconstruction.
    """

    _include_attrs = None
    _exclude_attrs = [
        'submodels',
        'data',
    ]

    class Node(object):
        def __init__(self, obj, parent):
            self.obj = obj
            self.parent = parent
            self.name = u'{}.{}'.format(parent.name, obj.name) if parent else obj.name

    def __init__(self, include_attrs=None, exclude_attrs=None, *args, **kwargs):
        super(ModelProperties, self).__init__()
        self.include_attrs = include_attrs or self._include_attrs
        self.exclude_attrs = exclude_attrs or self._exclude_attrs

    def to_model(self, obj_props):
        """
        Restore a global dictionary of model properties to a root model object and its submodels

        Input::

            {
                'root_name': <Name of root object>,
                'props': {
                    <model name>: {
                        <attribute name>: ...,
                        ...,
                        <attribute name>: ...
                    },
                    ...,
                    <model name>: {
                        <attribute name>: ...,
                        ...,
                        <attribute name>: ...
                    }
                }
            }

        Model names represent their nested structure like this:
            - <grandparent name>.<parent name>.<name>
        """
        objs_expanded = {}

        model_obj = root

        def _expand(name, data):
            #   Initialize the class with the original configuration, then set the saved attributes on it.
            obj = objs_expanded.get(name)
            if not obj:
                meta = data.pop('__meta')
                obj = meta['__class__'](config=data['orig_config'])
                for k, v in data.iteritems():
                    setattr(obj, k, v)
                #   Now associate expanded submodels
                submodel_names = getattr(obj, 'submodels', [])
                obj.submodels = [ _expand(name, obj_props.get(name)) for name in submodel_names ]
                objs_expanded[name] = obj
            return obj

        root_name = obj_props['root_name']
        root_props = obj_props['props'][root_name]
        root_obj = _expand(root_name, root_props)
        return root_obj


# @odo_convert.register(PICKLE, Model)
def from_model(model_obj, parent=None, incude_attrs=None, exclude_attrs=None):
    """
    Create a global dictionary of models and their properties.
    """
    node = self.Node(model_obj, parent)
    path_model_props = node.obj.path_model
    path_base, fname = path_model_props.rsplit(os.sep)
    # path_model =  TBD ???
    model_clone = node.obj.clone(
        node.obj,
        incude_attrs=include_attrs,
        exclude_attrs=exclude_attrs)
    obj_props = {
        'root': {
            'name': node.obj.name,
            'path': os.path.join(path_base, '.{}'.format(fname)),
        },
        'submodels': [
            self.from_model(submodel, node) for submodel in node.obj.submodels
        ]
    }
    return obj_props


    def clone(self, obj):

        cls = obj.__class__
        if self.include_attrs is not None:
            props = {k: getattr(self, k, None) for k, v in self.include_attrs}
        else:
            attrs = inspect.getmembers(obj)
            # No private attributes or routines
            attrs = filter(
                lambda x: not (x[0].startswith('__') or inspect.isroutine(x[1])),
                attrs)
            if self.exclude_attrs:
                #   Remove specifically excluded attrs
                attrs = filter(lambda x: x[0] not in set(self.exclude_attrs), attrs)
            props = dict(attrs)
        obj_new = cls
        return obj_new


# @odo_convert.register(ModelProperties, Model)
def model_to_modelproperties(model_obj, include_attrs=None, exclude_attrs=None, **kwargs):
    model_props_maker = ModelProperties(include_attrs=include_attrs, exclude_attrs=exclude_attrs)
    model_props = model_props_maker.compress(model_obj)
    return model_props


# @odo_convert.register(Model, ModelProperties)
def modelproperties_to_model(model_props, **kwargs):
    model_props_maker = ModelProperties()
    model_obj= model_props_maker.expand(model_props)
    return model_obj


# @odo_convert.register(ModelProperties, odo_util.PICKLE)
def pickle_to_modelproperties(ctx, **kwargs):
    pass


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
