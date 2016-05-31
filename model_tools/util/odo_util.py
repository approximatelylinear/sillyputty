#   Stdlib
import os
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
from odo import convert, append, resource
from odo import S3
from odo.utils import tmpfile

#   Types


class PICKLE(object):
    _canonical_extension = '.pkl'

    def __init__(self, path, *args, **kwargs):
        self.path = path


class MATRIXMARKET(object):
    _canonical_extension = '.mm'

    def __init__(self, path, *args, **kwargs):
        self.path = path


class YAML(object):
    _canonical_extension = '.yml'

    def __init__(self, path, *args, **kwargs):
        self.path = path


#   Resources
@resource.register('.*\.pkl')
def resource_pickle(uri, **kwargs):
    return PICKLE(uri, **kwargs)


@resource.register('.*\.mm')
def resource_mm(uri, **kwargs):
    return MATRIXMARKET(uri, **kwargs)


@resource.register('s3://.*\.pkl(\.gz)?', priority=15)
def resource_s3_pickle(uri, **kwargs):
    return S3(PICKLE)(uri, **kwargs)


@resource.register('.*\.(yml|yaml)')
def resource_yaml(uri, **kwargs):
    return YAML(uri, **kwargs)


#   Append operations
@append.register(PICKLE, object)
def append_object_to_pickle(c, data, **kwargs):
    with open(c.path, 'wb') as f:
        try:
            pickle.dump(data, f)
        except Exception as exc:
            raise

            # TBD
            try:
                cloudpickle.dump(data, f)
            except Exception as exc2:
                raise
    return c


@append.register(MATRIXMARKET, spmatrix)
def append_spmatrix_to_mm(c, data, **kwargs):
    with open(c.path, 'wb') as f:
        mmwrite(f, data)
    return c


@append.register(csc_matrix, np.ndarray)
def append_dense_to_csc(arr_csc, arr, **kwargs):
    return sparse_hstack((arr_csc, csc_matrix(arr)))


@append.register(YAML, object)
def append_object_to_yaml(c, data, **kwargs):
    with open(c.path, 'wb') as f_out:
        yaml.dump(data, f_out)
    return c


#   Convert operations
@convert.register(object, YAML)
def convert_yaml_to_object(c, **kwargs):
    with open(c.path, 'rbU') as f_in:
        return yaml.load(f_in)


@convert.register(coo_matrix, MATRIXMARKET)
def convert_mm_to_coo(c, **kwargs):
    with open(c.path, 'rb') as f_in:
        return mmread(f_in)


@convert.register(csr_matrix, MATRIXMARKET)
def convert_mm_to_csr(c, **kwargs):
    return convert(coo_matrix, c).tocsr()


@convert.register(csc_matrix, MATRIXMARKET)
def convert_mm_to_csc(c, **kwargs):
    return convert(coo_matrix, c).tocsc()


#   Examples
def examples():
    #   Yaml
    yaml_input = {'a': 1, 'b': 2}
    #       Test
    append(resource('test.yml'), yaml_input)
    assert os.path.exists('test.yml')
    yaml_load = convert(object, resource('test.yml'))
    assert yaml_input == yaml_load
    yaml_load2 = odo.odo('test.yml', object)
    assert yaml_input == yaml_load2
    #   !!! Fails !!!
    #       odo.odo({'a': 1, 'b': 2}, 'test2.yml')

    #   Sparse
    mat = np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], -1)
    append(resource('test.mm'), csc_matrix(mat))
    mat_out_coo = convert(coo_matrix, resource('test.mm'))
    #       Test
    assert (mat_out_coo.todense() == mat).all()
    print(type(mat_out_coo))
    mat_out_csr = convert(csr_matrix, resource('test.mm'))
    assert (mat_out_csr.todense() == mat).all()
    print(type(mat_out_csr))
    mat_out_csc = convert(csc_matrix, resource('test.mm'))
    assert (mat_out_csc.todense() == mat).all()
    print(type(mat_out_csc))
