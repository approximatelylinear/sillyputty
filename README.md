# Model Tools #

This provides helpers for implementing the Scikit-Learn interface used internally by Hume.

The overall rationale is to provide utilities for: 

1. Organizing model variants, by separating model definitions from their implementations.
2. Wrapping non-Scikit-learn code into the Scikit-Learn interface.
3. Persisting model definitions and data
3. Persisting model output

 
### What is this repository for? ###

* Helps ensure model methods correspond to the Scikit-Learn interface.
* Version 0.1.1

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* MJ Berends (mj@dose.com)
* Dose Data Science team


### Example Model Configuration

The three main components of a model definition are the `parameters`, the `wrapper_class` and the `model_callable`. 

- *Parameters*. These are the parameters passed to the `model_callable` as keyword arguments. Typically, you would always provide custom definitions.

```
# Custom `parameters`

models:
    vectorizer: &vectorizer
        name: vectorizer
        wrapper_class:
            module: 'model_tools.model'
            func: 'Model'
        model_callable:
            module: 'sklearn.feature_extraction.text'
            func: 'TfidfVectorizer'
        parameters:
            max_df: .9
            max_features: 2000000
            min_df: 2
            stop_words: null
            token_pattern: '(?u)\|'

#   Main entrypoint
model:
    *vectorizer
```

- ​*Model wrapper class*​. This class governs how model persistence, parameter parsing, and submodel definitions. For example, let’s say you just want to use an out-of-the box Scikit-learn model, but with some custom logic for manipulating the parameters. You would just subclass `Model` with your changes, as with the `Vectorizer` example below. The model callable in this case would be the vanilla scikit-learn class. 

   - There are two methods of defining a `wrapper_class`. 

     1. String. The class is a name in the namespace passed to the model runner.
     2. Dictionary with `module` and `func` keys. The class will be imported as `from <module> import <func>`. 

```
# Custom `wrapper_class`

models:
    vectorizer: &vectorizer
        name: vectorizer
        wrapper_class: Vectorizer
        model_callable:
            module: 'sklearn.feature_extraction.text'
            func: 'TfidfVectorizer'
        parameters:
            max_df: .9
            max_features: 2000000
            min_df: 2
            stop_words: null
            token_pattern: '(?u)\|'

#   Main entrypoint
model:
    *vectorizer
```

- ​*Model callable*​. If you want very fine-grained control over the model logic, you can create a custom model callable that is called by the generic `Model` class, as with the `AnnoyBuilder` example to follow.

   - There are two methods of defining a `model_callable`. 

     1. String. The callable is a name in the namespace passed to the model runner.
     2. Dictionary with `module` and `func` keys. The callable will be imported as `from <module> import <func>`. 

```
# Custom `model_callable`

models:
    neighbor_finder: &neighbor_finder
        name: neighbor_finder
        wrapper_class:
            module: 'model_tools.model'
            func: 'Model'
        model_callable: AnnoyBuilder
        parameters:
            n_features: 300
            n_trees: 100
            n_neighbors: 10
            search_k: 100000

#   Main entrypoint
model:
    *neighbor_finder
```

### Example Usage


```
import yaml
from toolz import memoize
from model_tools.model import Model
from model_tools.api import Runner

# See 'Example Model Configuration' above
PATH_MODEL_CONFIG = '/path/to/model/configuration.yml'


# Example model wrapper definition 
class Vectorizer(Model):
    _name = 'vectorizer'

    def _init_parameters(self, parameters):
        import re
        parameters = super(Vectorizer, self)._init_parameters(parameters)
        name = self.name
        model_parameters = parameters[name]
        token_pattern = model_parameters.pop('token_pattern', None)
        if token_pattern:
            model_parameters.setdefault('tokenizer',
                                        lambda x: re.split(token_pattern, x))
        return parameters


@memoize
def get_model_runner():
    with open(PATH_MODEL_CONFIG, 'rbU') as f_in:
        model_config = yaml.load(f_in)
        # `namespace=globals()` indicates that the model configuration refers to an object
        # defined in (or imported into) the current module.
        # 
        # Note that the `wrapper_class` defined in the configuration refers to a string 'Vectorizer'. 
        # Since no package is given, this is assumed to be an entry in `namespace`, which in this case
        # means it's a name in the current module.
        runner = Runner(MODEL_CONFIG, namespace=globals())
        return runner


#   Fit the model to some data. (This will persist any changes to the model to disk.)
get_model_runner().fit(some_data)

#   Transform new data with the fitted model. (This will also persist any changes to the model to disk.)
get_model_runner().transform(other_data)

```

### Example adapting non-Scikit-learn code to this interface

```
# Imports
#   Stdlib
import os
import logging
import pdb

#   3rd party
import numpy as np
import pandas as pd
from annoy import AnnoyIndex

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.path.abspath(os.getcwd())

LOGGER = logging.getLogger(__name__)
PATH_DATA = os.getenv('PATH_DATA') or os.path.join(THIS_DIR, 'data')

# Example config
"""
neighbor_finder: &neighbor_finder
    name: neighbor_finder
    wrapper_class:
        module: 'model_tools.model'
        func: 'Model'
    model_callable: AnnoyBuilder
    parameters:
        n_features: 300
        n_trees: 100
        n_neighbors: 10
        search_k: 100000

"""

class AnnoyBuilder(object):
    """
    Find Nearest Neighbors with ANNoy
    """

    #   Defaults
    _path = os.path.join(PATH_DATA, 'ix.ann')
    _n_features = None
    _n_trees = 100
    _n_neighbors = 10 # Default number to return
    _search_k = int(1e5)

    def __init__(self, path=None, **params):
        params.setdefault('n_features', self._n_features)
        params.setdefault('n_trees', self._n_trees)
        params.setdefault('n_neighbors', self._n_neighbors)
        params.setdefault('search_k', self._search_k)
        self.params = params
        self.path = path or self._path
        self.path_id_dictionary = self.path + '.iddict'

    def save(self):
        try:
            ix_annoy = self.index
        except AttributeError:
            LOGGER.exception("Must fit model before calling `save`.")
            raise
        else:
            ix_annoy.save(self.path)
            if not self.id_dictionary.empty:
                self.id_dictionary.to_csv(self.path_id_dictionary)

    def load(self):
        ix_annoy = AnnoyIndex(self.params['n_features'])
        ix_annoy.load(self.path)
        self.index = ix_annoy
        if os.path.exists(self.path_id_dictionary):
            self.id_dictionary = pd.read_csv(self.path_id_dictionary, index_col=0, header=None)
        else:
            self.id_dictionary = None

    def fit_sequence_of_data(self, Xs, **_ignore):
        params = self.params
        ix_annoy = AnnoyIndex(params['n_features'])
        id_dictionary = {}
        integer_id = 0
        for X in Xs:
            rows = (
                ((idx, vec.values) for idx, vec in X.iterrows()) if isinstance(X, pd.DataFrame)
                else enumerate(X))
            for idx, vec in rows:
                if not isinstance(idx, int):
                    id_dictionary[integer_id] = idx
                    idx = integer_id
                    integer_id += 1
                if idx % 1000 == 0:
                    LOGGER.debug(u"[ {} ] {} ".format(idx, vec))
                ix_annoy.add_item(idx, vec)
        ix_annoy.build(params['n_trees'])
        self.index = ix_annoy
        self.id_dictionary = pd.Series(id_dictionary)

    def fit(self, X, **_ignore):
        params = self.params
        ix_annoy = AnnoyIndex(params['n_features'])
        rows = (
            ((idx, vec.values) for idx, vec in X.iterrows()) if isinstance(X, pd.DataFrame)
            else enumerate(X))
        id_dictionary = {}
        integer_id = 0
        for idx, vec in rows:
            if not isinstance(idx, int):
                id_dictionary[integer_id] = idx
                idx = integer_id
                integer_id += 1
            if idx % 1000 == 0:
                LOGGER.debug(u"[ {} ] {} ".format(idx, vec))
            ix_annoy.add_item(idx, vec)
        ix_annoy.build(params['n_trees'])
        self.index = ix_annoy
        self.id_dictionary = pd.Series(id_dictionary)

    def predict(self, X, n_neighbors=None, search_k=None):
        n_neighbors = n_neighbors or self.params.get('n_neighbors')
        search_k = search_k or self.params.get('search_k')

        try:
            ix_annoy = self.index
        except AttributeError:
            LOGGER.exception("Must fit model before calling `predict`.")
            raise

        def _predict(idx, vec):
            idxs_annoy, dists_annoy = ix_annoy.get_nns_by_vector(
                vec, n_neighbors, search_k=search_k, include_distances=True)
            return pd.DataFrame({
                'source': [idx] * n_neighbors,
                'target': idxs_annoy,
                'dist': dists_annoy})

        if isinstance(X, pd.Series):
            rows = [(X.name, X.values)]
        elif isinstance(X, pd.DataFrame):
            rows = ((idx, vec.values) for idx, vec in X.iterrows())
        elif isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                rows = [(0, X)]
            else:
                rows = enumerate(X)
        elif isinstance(X, tuple):
            rows = [X]
        nns = [_predict(idx, vec) for idx, vec in rows]
        df = pd.concat(nns)
        if self.id_dictionary is not None and (not self.id_dictionary.empty):
            df['target'] = self.id_dictionary.loc[df['target']].values
        return df

    def fit_predict(self, X, n_neighbors=None, search_k=None):
        self.fit(X)
        X_pred = self.predict(
            X, n_neighbors=n_neighbors, search_k=search_k)
        return X_pred

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X):
        raise NotImplementedError
```
