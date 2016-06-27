
"""
Base classes
"""

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
import cloudpickle  # For function serialization
from sklearn.pipeline import make_pipeline as sk_make_pipeline

#   Custom current
from .exceptions import *
from .util.util import md5_hex, import_func
from .util import odo_util
from .util.odo_util import (
    odo, convert as odo_convert,
    resource as odo_resource, append as odo_append)

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.path.abspath(os.getcwd())

LOGGER = logging.getLogger(__name__)


class ModelFactory(object):
    _default_model_wrapper = Model

    def __init__(self, namespace=None, default_model_wrapper=None, *args, **kwargs):
        self.namespace = namespace or {}
        self.default_model_wrapper = self._default_model_wrapper if default_model_wrapper is None else default_model_wrapper

    def _init_config(self):
        #   TBD
        raise NotImplementedError

    def _load_attribute(self, info, namespace):
        """
        Load a class (or anything else really) given its name or location in package-space.

        TBD: Use multiple dispatch for this?
        """
        if isinstance(info, basestring):
            #   We just have the name of the function
            cls = namespace.get(info)
            if cls is None:
                #   Try to import as an attribute of the caller's module name
                cls = import_func(
                    module=namespace['__meta']['caller__name__'],
                    func=info)
        elif isinstance(info, dict):
            cls = import_func(**info)
        else:
            cls = None
        return cls

    def _init_submodels(self, namespace=None):
        """
        Create submodels
        """
        config = self.config

        submodels =
        return submodels

    def _init_model(self, namespace=None):
        """
        Create this model by instantiating the model callable with the parameters.
        """
        if globals_ is None:
            globals_ = globals()
        config = self.config
        submodels = self.submodels
        parameters = config.get('parameters') or {}
        model_callable = self._load_model_callable(config, globals_=globals_)
        model = None
        if not model_callable:
            if submodels is not None and len(submodels) == 1:
                submodel = submodels[0]
                if submodel is not None:
                    model = submodel
                # else:
                    # raise Exception(
                    #     "Could not find a model callable for [ {} ]".format(
                    #         self.__class__.__name__))
        else:
            model = model_callable(**parameters)
        return model

    def _make_model(self, config, namespace):
        params = {
            'config_str': yaml.dump(config),
            'name': config.get('name'),
            'namespace': namespace,
            'parameters': config.get('parameters'),
            'model_callable': self._load_model_callable(config, namespace),
            'submodels': self._load_submodel_callables(config, namespace),
            'path_model': (config.get('persistence') or {}).get('model'),
            'path_data': (config.get('persistence') or {}).get('data'),
        }
        wrapper_cls = self._load_model_wrapper(config, namespace=namespace)
        return wrapper_cls(**params)

    def _load_model_wrapper(self, config, namespace):
        info = config.get('wrapper_class')
        return info if isinstance(info, callable) else (self.default_model_wrapper if info is None else self._load_attribute(info, namespace))

    def _load_submodels(self, config, namespace):
        submodel_configs = config.get('model')
        return (
            [ self._make_model(v, namespace) for v in submodel_configs ]
            if submodel_configs else None)

    def _load_model_callable(self, config, namespace):
        info = config.get('model_callable')
        if info is not None:
            model_callable = info if isinstance(info, callable) else self._load_attribute(info, namespace)
            assert callable(model_callable)
        else:
            model_callable = None
        return model_callable
        #   Is there a place where I need this?
        # return getattr(self, '_model_callable', None) or load_model_callable(self.config, namespace)

    def __call__(self, config, namespace=None):
        if namespace is None:
            namespace = self.namespace or {}
        namespace = copy.copy(namespace)
        #   Keep track of the caller's module name to use as a fallback namespace
        namespace.setdefault('__meta', {})['caller__name__'] = inspect.stack()[1][0].f_globals['__name__']
        return self._make_model(config, namespace)


#   Instantiate callable that returns model instances
make_model = ModelFactory()


# def make_model(config, namespace=None):
#     if namespace is None:
#         namespace = globals()
#     params = {
#         'config': config,
#         'name': config.get('name'),
#         'namespace': namespace}
#     wrapper_cls = load_model_wrapper(config, namespace=namespace)
#     return wrapper_cls(**params)


# def load_model_wrapper(config, namespace=None):
#     info = config.get('wrapper_class')
#     return info if isinstance(info, callable) else (Model if info is None else load_attribute(info, namespace))


# def load_model_callable(config, namespace=None):
#     if namespace is None:
#         namespace = globals()
#     info = config.get('model_callable') or getattr(
#         self, '_model_callable', None)
#     if info is not None:
#         model_callable = info if isinstance(info, callable) else get_attribute(info, namespace)
#         assert callable(model_callable)
#     return model_callable


# class Configurer(object):
#     def __init__(self, config=None, *args, **kwargs):
#         super(Configurer, self).__init__()
#         self.config = self._init_config() if config is None else config

#     def _init_config(self):
#         return None

    # def _setup_persister(self, config, globals_=None):
    #     def _get_persister_class(config, globals_):
    #         namespace = {
    #             'pickle': PicklerMixin,
    #             's3': S3Mixin,
    #             'sqs': SQSMixin,
    #         }
    #         cls = config.get('persister') or 'pickle'
    #         cls = load_attribute(cls, globals_, namespace)
    #         return cls

    #     persister_cls = _get_persister_class(config, globals_)
    #     return persister_cls(**config['persister_params'])s


    # def _load_model_callable(self, namespace=None):
    #     return getattr(self, '_model_callable', None) or load_model_callable(self.config, namespace)





class Persister(object):
    _persist_attrs = None

    def __init__(self, *args, **kwargs):
        super(Persister, self).__init__()

    def fltr_attrs(self):
        if self._persist_attrs is not None:
            obj = {k: getattr(self, k, None) for k, v in self._persist_attrs}
        else:
            fltr = lambda x: not (x[0].startswith('__') or inspect.isroutine(x[1]))
            attrs = inspect.getmembers(self)
            obj = dict(filter(fltr, attrs))
        return obj

    def _save(self, obj):
        raise NotImplementedError

    def _load(self):
        raise NotImplementedError

    def save(self):
        obj = self.fltr_attrs()
        self._save(obj)
        return obj

    def load(self):
        obj = self._load()
        for k, v in obj.iteritems():
            setattr(self, k, v)
        return self


class PicklerMixin(Persister):
    def __init__(self, path_model, *args, **kwargs):
        super(PicklerMixin, self).__init__(*args, **kwargs)
        self.path_model = path_model

    def _save(self, obj):
        with open(self.path_model, 'wb') as f_out:
            cloudpickle.dump(obj, f_out)

    def _load(self):
        if os.path.exists(self.path_model):
            with open(self.path_model, 'rb') as f_in:
                obj = pickle.load(f_in)
                return obj


class S3Mixin(Persister):
    import boto3

    def __init__(self, path_model, *args, **kwargs):
        super(S3Mixin, self).__init__(*args, **kwargs)
        self.path_model = path_model
        client = self.boto3.client('s3')
        self.client = client


class SQSMixin(Persister):
    import boto3

    def __init__(self, path_model, *args, **kwargs):
        super(SQSMixin, self).__init__(*args, **kwargs)
        self.path_model = path_model
        client = self.boto3.client('sqs')
        self.client = client


# class Model(Configurer):
class Model(object):
    """
    Wrapper for a model that handles configuration and provides entrypoints.  It implements the scikit-learn model interface.

    Attributes
        name: Model identifier
        stats: Model run time stats
        config: Model configuration
        submodels: Submodels created and called by this model
        parameters: Dictionary of parameters for this model and its submodels, keyed by model name.
        id: MD5 hash of `repr` of model parameters
        data: Data generated by this model.

    """
    _model_callable = None
    _name = None

    @staticmethod
    def clone(obj, include_attrs=None, exclude_attrs=None):
        """
        Create a new object containing relevant attributes and values from an input object (assumed to be a (sub-)instance of `Model`.)
        """
        _include_attrs = None
        _exclude_attrs = set([
            'submodels',
            'data',
        ])
        if include_attrs is None:
            include_attrs = _include_attrs
        if exclude_attrs is None:
            exclude_attrs = _exclude_attrs
        cls = obj.__class__
        if include_attrs:
            props = {k: getattr(obj, k, None) for k in include_attrs}
        else:
            attrs = inspect.getmembers(obj)
            # No private attributes or routines
            attrs = filter(
                lambda x: not (x[0].startswith('__') or inspect.isroutine(x[1])),
                attrs)
            if exclude_attrs:
                #   Remove specifically excluded attrs
                attrs = filter(lambda x: x[0] not in set(exclude_attrs), attrs)
            props = dict(attrs)
        obj_new = cls(_is_clone=True)
        for k, v in props.iteritems():
            setattr(obj_new, k, v)
        return obj_new

    def __init__(self, name=None, model_callable=None, parameters=None, submodels=None, namespace=None, path_model=None, path_data=None, config_str=None, _is_clone=None, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.data = {}
        self.stats = {}
        if not _is_clone:
            self.model_callable = self._model_callable if model_callable is None else model_callable
            self.name = name or self._name
            self.config_str = config_str
            self.path_model = path_model
            self.path_data = path_data
            self.parameters = self._init_parameters(copy.deepcopy(parameters) or {})
            self.model = self._init_model()
            self.id = self._init_id()

    def _init_parameters(self, parameters):
        config = self.config
        submodels = self.submodels
        parameters = {}
        parameters[self.name] = parameters
        if submodels is not None:
            for submodel in submodels:
                parameters[submodel.name] = submodel.parameters
        return parameters

    def update_parameters(self):
        """
        The parameters might change during fitting or via grid search.
        TBD"""

    def _init_id(self):
        """
        md5 hex of the string representation of the parameters
        """
        parameters = self.parameters
        parameters_str = repr(parameters)
        parameters_md5 = md5_hex(parameters_str)
        return parameters_md5

    def _init_model(self):
        """
        Create this model by instantiating the model callable with the parameters.
        """
        model = None
        parameters = self.parameters
        model_callable = self.model_callable
        if not model_callable:
            submodels = self.submodels
            if submodels is not None and len(submodels) == 1:
                submodel = submodels[0]
                if submodel is not None:
                    #   TBD: Should this be `submodel` or `submodel.model`?
                    model = submodel.model
                # else:
                    # raise Exception(
                    #     "Could not find a model callable for [ {} ]".format(
                    #         self.__class__.__name__))
        else:
            model = model_callable(**parameters)
        return model

    def save(self):
        return self.persister.save()

    def load(self):
        return self.persister.load()

    def fit(self, X, **kwargs):
        """
        Delegate to fit method on `self.model`.
        """
        self.model.fit(X, **kwargs)

    def transform(self, X, **kwargs):
        """
        Delegate to the `self.model`'s `transform` method and put the result in the `X_transform` inside the `data` attribute.
        """
        X_trans = self.model.transform(X, **kwargs)
        self.data['X_transform'] = X_trans
        return X_trans

    def fit_transform(self, X, **kwargs):
        """
        Delegate to the `self.model`'s `fit_transform` method and put the result in the `X_transform` inside the `data` attribute.
        """
        X_trans = self.model.fit_transform(X, **kwargs)
        self.data['X_transform'] = X_trans
        return X_trans

    def predict(self, X, **kwargs):
        """
        Delegate to the `self.model`'s `predict` method and put the result in the `y_pred` inside the `data` attribute.
        """
        y_pred = self.model.predict(X, **kwargs)
        self.data['y_pred'] = y_pred
        return y_pred

    def fit_predict(self, X, **kwargs):
        """
        Delegate to the `self.model`'s `fit_predict` method and put the result in the `y_pred` inside the `data` attribute.
        """
        y_pred = self.model.fit_predict(X, **kwargs)
        self.data['y_pred'] = y_pred
        return y_pred

    def __getattr__(self, name):
        """
        Delegate to properties of `self.model`.
        """
        if hasattr(self, 'model'):
            try:
                val = getattr(self.model, name)
            except AttributeError:
                raise
            else:
                return val
        else:
            raise AttributeError(name)



class Pipeliner(Model):
    _model_callable = lambda x: None
    _name = 'pipeliner'

    def __init__(self, *args, **kwargs):
        super(Pipeliner, self).__init__(*args, **kwargs)

    def _init_model(self, globals_=None):
        if self.submodels:
            return self.submodels[-1]

    def fit(self, X):
        """
        Fit and transform all models up until the last one, which we just fit.
        """
        X_trans = X
        for submodel in self.submodels[:-1]:
            LOGGER.debug("Fitting and transforming with [ {} ]".format(
                submodel.name))
            X_trans = submodel.fit_transform(X_trans)
        self.data['X_fit'] = X_trans
        #   Now just perform a fit on the last model.
        self.submodels[-1].fit(X_trans)

    def fit_transform(self, X):
        """
        Fit and transform all submodels
        """
        X_trans = X
        for submodel in self.submodels:
            LOGGER.debug("Fitting and transforming with [ {} ]".format(
                submodel.name))
            X_trans = submodel.fit_transform(X_trans)
        # Apply
        self.data['X_transform'] = X_trans
        return X_trans

    def transform(self, X):
        X_trans = X
        for submodel in self.submodels:
            LOGGER.debug("Transforming with [ {} ]".format(submodel.name))
            X_trans = submodel.transform(X_trans)
        self.data['X_transform'] = X_trans
        return X_trans

    def predict(self, X):
        X_trans = X
        for submodel in self.submodels[:-1]:
            LOGGER.debug("Transforming with [ {} ]".format(submodel.name))
            X_trans = submodel.transform(X_trans)
        y_pred = self.submodels[-1].predict(X_trans)
        self.data['y_pred'] = y_pred
        return y_pred

    def get_labels(self):
        model = self.model
        if hasattr(model.model, 'labels_'):
            return model.model.labels_

    def __getattr__(self, name):
        """
        Delegate to properties of `self.model`.
        """
        if hasattr(self, 'model'):
            try:
                val = getattr(self.model, name)
            except AttributeError:
                try:
                    val = getattr(self.model.model, name)
                except AttributeError:
                    raise
                else:
                    return val
            else:
                return val
        else:
            raise AttributeError(name)



class SKPipeliner(Model):
    _model_callable = sk_make_pipeline
    _name = 'pipeliner_sk'

    def __init__(self, *args, **kwargs):
        super(SKPipeliner, self).__init__(*args, **kwargs)

    def _init_model(self, globals_=None):
        if globals_ is None:
            globals_ = globals()
        name = self.name
        config = self.config
        submodels = self.submodels
        model_callable = self._load_model_callable(config, globals_=globals_)
        if not model_callable:
            raise Exception(
                "Could not find a model callable for [ {} ]".format(
                    self.__class__.__name__))
        else:
            model = model_callable(*[sm.model for sm in submodels])
        return model
