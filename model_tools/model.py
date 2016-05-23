#   Stdlib
import os
import re
import gzip
from time import time
import pdb
from pprint import pformat
import logging
import copy
import textwrap
import inspect
try:
    import cPickle as pickle
except ImportError:
    import pickle

#   3rd party
import yaml
import numpy as np
import pandas as pd
import cloudpickle  # For function serialization
from sklearn.pipeline import make_pipeline as sk_make_pipeline

#   Custom current
from .exceptions import *
from .util.util import md5_hex, import_func

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.path.abspath(os.getcwd())

LOGGER = logging.getLogger(__name__)


def make_model(config, globals_=None):
    if globals_ is None:
        globals_ = globals()
    wrapper_cls = config.get('wrapper_class')
    if wrapper_cls is None:
        wrapper_cls = Model
    else:
        if isinstance(wrapper_cls, basestring):
            #   It's a string.
            wrapper_cls = globals_.get(wrapper_cls)
        elif isinstance(wrapper_cls, dict):
            wrapper_cls = import_func(**wrapper_cls)
    """
    model = wrapper_cls(config=config, name=config.get('name'), globals_=globals_)
    """
    model = wrapper_cls(config=config,
                        name=config.get('name'),
                        globals_=globals_)
    return model


class Configurer(object):
    def __init__(self, config=None, *args, **kwargs):
        super(Configurer, self).__init__()
        self.config = self._init_config() if config is None else config

    def _init_config(self):
        return None

    def _load_model_wrapper(self, config, globals_=None):
        if globals_ is None:
            globals_ = globals()
        name = config['name']
        wrapper_cls = config['wrapper_class']
        if isinstance(wrapper_cls, basestring):
            #   It's a string.
            wrapper_cls = globals_.get(wrapper_cls)
        elif isinstance(wrapper_cls, dict):
            wrapper_cls = import_func(**wrapper_cls)
        submodel = wrapper_cls(config=config, name=name)
        return submodel

    def _load_model_callable(self, config, globals_=None):
        if globals_ is None:
            globals_ = globals()
        config = self.config
        model_callable = config.get('model_callable') or getattr(
            self, '_model_callable', None)
        if model_callable is not None:
            #   Does it already exist in the global namespace?
            if isinstance(model_callable, basestring):
                model_callable = globals_.get(model_callable)
            elif isinstance(model_callable, dict):
                model_callable = import_func(**model_callable)
            else:
                assert callable(model_callable)
        return model_callable


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
            cloudpicklepickle.dump(obj, f_out)

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


class Model(Configurer):
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

    def __init__(self, name=None, globals_=None, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        if globals_ is None:
            globals_ = globals()
        self.name = self._name or name
        self.stats = {}
        self.submodels = self._init_submodels(globals_=globals_)
        self.parameters = self._init_parameters()
        self.model = self._init_model(globals_=globals_)
        self.id = self._init_id()
        self.data = {}

    def _init_parameters(self):
        config = self.config
        submodels = self.submodels
        parameters = {}
        parameters[self.name] = config.get('parameters') or {}
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

    def _init_submodels(self, globals_=None):
        """
        Create submodels
        """
        if globals_ is None:
            globals_ = globals()
        config = self.config
        submodel_configs = config.get('model')
        submodels = [self._load_model_wrapper(v, globals_)
                     for v in submodel_configs] if submodel_configs else None
        return submodels

    def _init_model(self, globals_=None):
        """
        Create this model by instantiating the model callable with the parameters.
        """
        if globals_ is None:
            globals_ = globals()
        config = self.config
        submodels = self.submodels
        parameters = config.get('parameters') or {}
        model_callable = self._load_model_callable(config, globals_=globals_)
        if not model_callable:
            if submodels is not None and len(submodels) == 1:
                submodel = submodels[0]
                if submodel is not None:
                    model = submodel
                else:
                    raise Exception(
                        "Could not find a model callable for [ {} ]".format(
                            self.__class__.__name__))
        else:
            model = model_callable(**parameters)
        return model

    def fit(self, X):
        """
        Delegate to fit method on `self.model`.
        """
        self.model.fit(X)

    def transform(self, X):
        """
        Delegate to the `self.model`'s `transform` method and put the result in the `X_transform` inside the `data` attribute.
        """
        X_trans = self.model.transform(X)
        self.data['X_transform'] = X_trans
        return X_trans

    def fit_transform(self, X):
        """
        Delegate to the `self.model`'s `fit_transform` method and put the result in the `X_transform` inside the `data` attribute.
        """
        X_trans = self.model.fit_transform(X)
        self.data['X_transform'] = X_trans
        return X_trans

    def predict(self, X):
        """
        Delegate to the `self.model`'s `predict` method and put the result in the `y_pred` inside the `data` attribute.
        """
        y_pred = self.model.predict(X)
        self.data['y_pred'] = y_pred
        return y_pred

    def __getattr__(self, name):
        """
        Delegate to properties of `self.model`.
        """
        # LOGGER.debug("Getting {}".format(name))
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
        X_trans = X
        for submodel in self.submodels[:-1]:
            # Fit and transform all models up until the last one.
            LOGGER.debug("Fitting and transforming with [ {} ]".format(
                submodel.name))
            X_trans = submodel.fit_transform(X_trans)
        self.data['X_fit'] = X_trans
        #   Now just perform a fit on the last model.
        self.submodels[-1].fit(X_trans)

    def fit_transform(self, X):
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
        # pdb.set_trace()
        # LOGGER.debug("Getting {}".format(name))
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
            model = model_callable(*[sm.model for sm in submodel])
        return model
