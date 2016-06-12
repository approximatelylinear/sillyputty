
#   Stdlib
import os
import urlparse
try:
    import cPickle as pickle
except ImportError:
    import pickle

#   3rd party
import yaml
import numpy as np
from scipy.sparse import spmatrix, csc_matrix, csr_matrix, coo_matrix, hstack as sparse_hstack
from scipy.io import mminfo, mmread, mmwrite
import odo
from odo import (
    convert as odo_convert, append as odo_append,
    resource as odo_resource, discover as odo_discover)
from odo import S3, Temp
from odo.utils import tmpfile

#   Types

class PICKLE(object):
    _canonical_extension = '.pkl'

    def __init__(self, path=None, data=None, *args, **kwargs):
        self.path = path
        self.data = data


class MATRIXMARKET(object):
    _canonical_extension = '.mm'

    def __init__(self, path=None, data=None, *args, **kwargs):
        self.path = path
        self.data = data


class YAML(object):
    _canonical_extension = '.yml'

    def __init__(self, path=None, data=None, *args, **kwargs):
        self.path = path
        self.data = data


#   Resources
@odo_resource.register(ur'.*\.pkl')
def resource_pickle(uri, **kwargs):
    return PICKLE(uri, **kwargs)


@odo_resource.register(ur'.*\.mm')
def resource_mm(uri, **kwargs):
    return MATRIXMARKET(uri, **kwargs)


@odo_resource.register(ur's3://.*\.pkl(\.gz)?', priority=15)
def resource_s3_pickle(uri, **kwargs):
    return S3(PICKLE)(uri, **kwargs)


@odo_resource.register(ur'.*\.(yml|yaml)')
def resource_yaml(uri, **kwargs):
    return YAML(uri, **kwargs)


#   Convert operations
@odo_convert.register(object, YAML)
def convert_yaml_to_object(c, **kwargs):
    with open(c.path, 'rbU') as f_in:
        return yaml.load(f_in)


@odo_convert.register(coo_matrix, MATRIXMARKET)
def convert_mm_to_coo(c, **kwargs):
    with open(c.path, 'rb') as f_in:
        return mmread(f_in)


@odo_convert.register(csr_matrix, MATRIXMARKET)
def convert_mm_to_csr(c, **kwargs):
    return odo_convert(coo_matrix, c).tocsr()


@odo_convert.register(csc_matrix, MATRIXMARKET)
def convert_mm_to_csc(c, **kwargs):
    return odo_convert(coo_matrix, c).tocsc()


@odo_convert.register(PICKLE, object)
def convert_object_to_pickle(data, **kwargs):
    try:
        data_pkl = pickle.dumps(data)
    except Exception as exc:
        raise
        # TBD
        # try:
        #     data_pkl = cloudpickle.dumps(data)
        # except Exception as exc2:
        #     raise
    return data_pkl


# @odo_convert.register(object, PICKLE)
# def convert_pickle_to_object(pkl_proxy):
#     parsed = urlparse.urlparse(pkl_proxy.path)
#     if parsed.scheme == 'memory':
#         obj = pickle.loads(pkl_proxy.data)
#     else:
#         with open(pkl_proxy.path, 'wb') as f_in:
#             obj = pickle.load(f_in)
#     return obj


# @odo_convert.register(object, S3(PICKLE))
# def convert_s3_pickle_to_object(s3_proxy, **kwargs):
#     return odo_convert(object, PICKLE(s3_proxy.object.get_contents_as_string()))


# #   Append operations
# @odo_append.register(PICKLE, object)
# def append_object_to_pickle(c, data, **kwargs):
#     obj_as_pkl = odo_convert(c, data)
#     parsed = urlparse.urlparse(c.path)
#     if parsed.scheme == 'memory':
#         c.data = obj_as_pkl
#     else:
#         with open(c.path, 'wb') as f_out:
#             f_out.write(obj_as_pkl)
#     return c


# @odo_append.register(S3(PICKLE), object)
# def anything_to_s3_pickle(s3_proxy, data, **kwargs):
#     with tmpfile('.pkl', '.') as fname:
#         with open(fname, 'wb') as f_out:
#             odo_append(PICKLE(fname), data)
#             s3_proxy.object.set_contents_from_filename(fname)
#     return s3_proxy


# @odo_append.register(PICKLE, S3(PICKLE))
# def s3_to_pickle(pkl_proxy, s3_proxy, **kwargs):
#     s3_proxy.object.get_contents_to_filename(pkl_proxy.path)
#     return s3_proxy.subtype(pkl_proxy.path, **kwargs)


@odo_append.register(MATRIXMARKET, spmatrix)
def append_spmatrix_to_mm(c, data, **kwargs):
    with open(c.path, 'wb') as f:
        mmwrite(f, data)
    return c


@odo_append.register(csc_matrix, np.ndarray)
def append_dense_to_csc(arr_csc, arr, **kwargs):
    return sparse_hstack((arr_csc, csc_matrix(arr)))


@odo_append.register(YAML, object)
def append_object_to_yaml(c, data, **kwargs):
    with open(c.path, 'wb') as f_out:
        yaml.dump(data, f_out)
    return c




#   Examples
def examples():
    #   Yaml
    yaml_input = {'a': 1, 'b': 2}
    #       Test
    odo_append(odo_resource('test.yml'), yaml_input)
    assert os.path.exists('test.yml')
    yaml_load = odo_convert(object, odo_resource('test.yml'))
    assert yaml_input == yaml_load
    yaml_load2 = odo.odo('test.yml', object)
    assert yaml_input == yaml_load2
    #   !!! Fails !!!
    #       odo.odo({'a': 1, 'b': 2}, 'test2.yml')

    #   Sparse
    mat = np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], -1)
    odo_append(odo_resource('test.mm'), csc_matrix(mat))
    mat_out_coo = odo_convert(coo_matrix, odo_resource('test.mm'))
    #       Test
    assert (mat_out_coo.todense() == mat).all()
    print(type(mat_out_coo))
    mat_out_csr = odo_convert(csr_matrix, odo_resource('test.mm'))
    assert (mat_out_csr.todense() == mat).all()
    print(type(mat_out_csr))
    mat_out_csc = odo_convert(csc_matrix, odo_resource('test.mm'))
    assert (mat_out_csc.todense() == mat).all()
    print(type(mat_out_csc))
