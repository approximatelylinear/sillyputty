

# Imports
#   Stdlib
import datetime
import time
import string
import os
import logging
import pdb
from pprint import pformat

#   3rd party
import numpy as np
import pandas as pd
import scipy as sp
from scipy.io import mmread
from scipy.sparse import csc_matrix, coo_matrix


LOGGER = logging.getLogger(__name__)


def to_h5(h5store, path_df, path_h5, read_csv_params=None, df=None):
    if read_csv_params is None:
        read_csv_params = {}
    if df is None:
        df = pd.read_csv(path_df, **read_csv_params)
    # Get version
    node = h5store.get_node(path_h5)
    if node is None:
        version = 1
    else:
        version = node._v_attrs['version'] + 1 if 'version' in node._v_attrs else 1
    h5store.put(path_h5, df, format='fixed')
    node = h5store.get_node(path_h5)
    node._v_attrs['version'] = version
    node._v_attrs['file_path'] = path_df
    node._v_attrs['shape'] = h5store[path_h5].shape
    node._v_attrs['timestamp'] = int(time.mktime(time.gmtime()))
    if '_id' not in node._v_attrs:
        node._v_attrs['_id'] = uuid.uuid5(uuid.NAMESPACE_URL, path_h5).hex
    h5store.flush()
    print(h5store[path_h5].info())
    print(repr(node._v_attrs))
    return node



def sparse_to_h5(h5store, path_S, path_h5, S=None):
    if S is None:
        S = mmread(path_S)
    # S must be in coo format
    df = pd.DataFrame({
        'row': S.row,
        'col': S.col,
        'data': S.data})
    node = to_h5(h5store, path_S, path_h5, df=df)
    node._v_attrs['sparse_shape'] = S.shape
    return node


def sparse_from_h5(h5store, path_h5):
    """
    """
    # Dataframe with data values, and corresponding row indices and column indices
    df = h5store[path_h5]
    node = h5store.get_node(path_h5)
    S = coo_matrix(
        (df['data'].values, (df['row'].values, df['col'].values)),
        shape=node._v_attrs['sparse_shape'])
    S = csc_matrix(S)
    return S


def from_h5(h5_store, path):
    df = h5_store[path]
    node = h5_store.get_node(path)
    df.h5_version = node._v_attrs['version']
    df.h5_file_path = node._v_attrs['file_path']
    df.h5_timestamp = node._v_attrs['timestamp']
    return df
