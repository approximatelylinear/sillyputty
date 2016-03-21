
#   Stdlib
import os
import hashlib
import re
import json
import logging
import pdb
import urlparse
import urllib
import itertools
from pprint import pformat
from io import BytesIO

#   3rd party
import requests
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file as svmlight_load, dump_svmlight_file as svmlight_dump

#   Custom
from .s3 import S3Adapter

LOGGER = logging.getLogger(__name__)


def md5_hex(val):
    """Retrieve the MD5 hash of a value as a hex string.
    """
    m = hashlib.md5()
    m.update(val)
    return m.hexdigest()


def import_func(module, func):
    _temp = __import__(module, globals(), locals(), [func], -1)
    return getattr(_temp, func)


def load_data(uri_or_data):
    """
    Possible uri formats:
        - <content_type>://<path>
            - Local file formatted according to the content type in the scheme
        - <protocol>://<path>
            - Remote file available at the given uri.
            - The response should specify the file's encoding in the 'content-type' header.

    """
    m = re.match(ur'(?P<scheme>\w+)://(?P<path>.+)', uri_or_data)
    if m:
        # It's a uri
        scheme = m.group('scheme')
        path = m.group('path')
        if scheme == 's3':
            #   File is located on s3
            sess = requests.Session()
            sess.mount('s3', S3Adapter())
            data = sess.get(uri_or_data)
        elif scheme == 'file':
            uri_or_data = path
            #   Get content type from extension
            _, ext = os.path.splitext(uri_or_data)
            if ext == '.json':
                content_type = 'json'
            elif ext == '.svmlight':
                content_type = 'svmlight'
            elif ext == '.csv':
                content_type = 'csv'
            data = None
        else:
            #   It's a remote resource.
            try:
                resp = requests.get(uri_or_data)
            except IOError as exc_io:
                raise
            else:
                try:
                    resp.raise_for_status()
                except Exception as exc_http:
                    raise
                else:
                    data = resp.content
                    #   Check the content type
                    content_type = resp.headers['content-type']
                    if 'json' in content_type:
                        content_type = 'json'
                    elif 'svmlight' in content_type:
                        content_type = 'svmlight'
                    elif 'csv' in content_type:
                        content_type = 'csv'
                    else:
                        raise Exception("Unknown content-type: [ {} ]".format(content_type))
    else:
        # Data was passed in.
        # TBD: Permit data stream with EOF indicated by '\n\n'
        uri_or_data = uri_or_data.decode('utf-8')
        if uri_or_data.startswith(u'{'):
            #   Treat it as json
            content_type = 'json'
            data = uri_or_data
        else:
            content_type = 'svmlight'
            data = uri_or_data
    data = load_by_content_type(content_type, uri_or_data=uri_or_data, data=data)
    return data


def load_by_content_type(content_type, uri_or_data=None, data=None):
    assert (uri_or_data is not None) or (data is not None)
    if isinstance(data, basestring):
        if content_type == 'json':
            data = json.loads(data)
        elif content_type == 'csv':
            data = csv_loads(data)
        elif content_type == 'svmlight':
            data = svmlight_loads(data)
    else:
        assert uri_or_data is not None
        with open(uri_or_data, 'rbU') as f_in:
            if content_type == 'json':
                try:
                    data = json.load(f_in)
                except Exception as exc_json:
                    raise
            elif content_type == 'csv':
                try:
                    data = pd.read_csv(f_in)
                except Exception as exc_csv:
                    raise
            elif content_type == 'svmlight':
                try:
                    data = svmlight_load(f_in)
                except Exception as exc_svmlight:
                    raise
    return data


def csv_loads(data):
    if isinstance(data, unicode):
        data = data.encode('utf-8')
    f_in = BytesIO(data)
    data_fmt = pd.read_csv(f_in)
    return data_fmt


def svmlight_loads(data):
    if isinstance(data, unicode):
        data = data.encode('utf-8')
    f_in = BytesIO(data)
    data_fmt = svmlight_load(f_in)
    return data_fmt


def test_load_data():
    if not os.path.exists('abc.json'):
        with open('abc.json', 'wb') as f_out:
            json.dump({'a': 'b', 'c': 'd'}, f_out)
    if not os.path.exists('abc.svmlight'):
        X = np.array([[1, 0, 1], [0, 1, 0]])
        LOGGER.info(X.shape)
        LOGGER.info(X)
        y = np.array([1, 2])
        LOGGER.info(y.shape)
        LOGGER.info(y)
        with open('abc.svmlight', 'wb') as f_out:
            svmlight_dump(X=X, y=y, f=f_out, comment='labels:b d features:a c e')
    if not os.path.exists('abc.csv'):
        with open('abc.csv', 'wb') as f_out:
            pd.DataFrame([{'a': 'b', 'c': 'd'}]).to_csv(f_out, index=False, header=True)
    uri_or_data_list = [
        'json://abc.json',
        'svmlight://abc.svmlight',
        'csv://abc.csv',
        '{"a": "b", "c": "d"}',
    ]
    for uri_or_data in uri_or_data_list:
        data = load_data(uri_or_data)
        LOGGER.info(data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_load_data()
