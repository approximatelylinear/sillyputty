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

LOGGER = logging.getLogger(__name__)


class BaseAdapter(object):
    """The Base Transport Adapter"""

    def __init__(self):
        super(BaseAdapter, self).__init__()

    def send(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class MissingBucket(Exception):
    pass


class S3ContentList(object):
    def __init__(self, items):
        super(S3ContentList, self).__init__()
        self.items = items

    def read(self, chunk_size=None):
        def _get():
            for item in self.items:
                if (isinstance(item, dict) and hasattr(
                        item.get('Body'), 'read')):
                    content = item['Body'].read(amt=chunk_size)
                    if len(content):
                        yield content
                else:
                    yield ''

        result = list(_get())
        if not any(result):
            result = []
        return result

    def close(self):
        for item in self.items:
            if (isinstance(item, dict) and hasattr(item.get('Body'), 'close')):
                item['Body'].close()


class S3Content(object):
    def __init__(self, item):
        super(S3Content, self).__init__()
        self.item = item

    def read(self, chunk_size=None):
        pdb.set_trace()
        item = self.item
        if (isinstance(item, dict) and hasattr(item.get('Body'), 'read')):
            content = item['Body'].read(amt=chunk_size)
        else:
            content = None
        return content

    def close(self):
        item = self.item
        if (isinstance(item, dict) and hasattr(item.get('Body'), 'close')):
            item['Body'].close()


class S3Dict(object):
    def __init__(self, item):
        super(S3Dict, self).__init__()
        #   TBD: What about items that are json strings?
        self.content = item


class S3DictList(object):
    def __init__(self, items):
        super(S3DictList, self).__init__()
        self.content = []
        #   TBD: What about items that are json strings?
        for item in items:
            if isinstance(item, dict):
                self.content.append(items)


class Response(object):
    """The :class:`Response <Response>` object, which contains a
    server's response to an HTTP request.
    """


class S3Response(requests.models.Response):

    __attrs__ = requests.models.Response.__attrs__ + ['parsed_json']

    def __init__(self, resp=None):
        super(S3Response, self).__init__()
        if resp is not None:
            # Fallback to None if there's no status_code, for whatever reason.
            self.status_code = getattr(resp, 'status', None)
            self.reason = getattr(resp, 'reason', None)
            content = getattr(resp, 'content', None)
            if content is not None:
                self._content = content
                if (isinstance(content, dict) or isinstance(content, list)):
                    self.parsed_json = content
            else:
                if hasattr(resp, 'read') and hasattr(resp, 'close'):
                    #   File-like object
                    self.raw = resp

    @property
    def parsed_json(self):
        try:
            self._parsed_json
        except AttributeError:
            self._parsed_json = None
        return self._parsed_json

    @parsed_json.setter
    def parsed_json(self, val):
        self._parsed_json = val

    def json(self, **kwargs):
        """Returns the json-encoded content of a response, if any.

        :param \*\*kwargs: Optional arguments that ``json.loads`` takes.
        """
        parsed_json = self.parsed_json
        if parsed_json is None:
            parsed_json = super(S3Response, self).json(**kwargs)
        return parsed_json

    @property
    def content(self):
        """Content of the response, in bytes."""

        if self._content is False:
            # Read the contents.
            try:
                if self._content_consumed:
                    raise RuntimeError(
                        'The content for this response was already consumed')

                if self.status_code == 0:
                    self._content = None
                else:
                    _content = list(self.iter_content(
                        requests.models.CONTENT_CHUNK_SIZE))
                    if isinstance(self.raw, S3ContentList):
                        #   Join the individual segments.
                        _content = list(itertools.izip(*_content))
                        self._content = [requests.compat.bytes().join(c)
                                         for c in _content]
                    else:
                        #   Join the chunks
                        self._content = requests.compat.bytes().join(
                            _content) or bytes()

            except AttributeError:
                self._content = None

        self._content_consumed = True
        # don't need to release the connection; that's been handled by urllib3
        # since we exhausted the data.
        return self._content


class S3Adapter(BaseAdapter):
    """
    Transport adapter for s3 stuff
    """
    import boto3
    import botocore

    def __init__(self):
        super(BaseAdapter, self).__init__()

    def build_response(self, req, resp):
        response = S3Response(resp=resp)
        if isinstance(req.url, bytes):
            response.url = req.url.decode('utf-8')
        else:
            response.url = req.url
        # Give the Response some context.
        response.request = req
        response.connection = self
        return response

    def get_connection(self):
        #   Make a boto3 connection
        s3 = self.boto3.resource('s3')
        return s3

    def get_bucket(self, s3, url):
        parsed = urlparse.urlparse(url)
        name = parsed.netloc
        bucket = s3.Bucket(name)
        try:
            s3.meta.client.head_bucket(Bucket=name)
        except self.botocore.exceptions.ClientError as exc:
            # If a client error is thrown, then check that it was a 404 error.
            # If it was a 404 error, then the bucket does not exist.
            error_code = int(exc.response['Error']['Code'])
            if error_code == 404:
                raise MissingBucket
        return bucket

    def do_get(self, url):
        """
        s3://bucket_bar/key_foo
            #   Get bucket_bar.key_foo
        s3://bucket_bar/key_foo/_download_file
            #   Download file at bucket_bar.key_foo to key_foo
        s3://bucket_bar
            #   Get all keys
        s3://bucket_bar/_all/_download_file
            #   Download files at all keys in bucket_bar
        s3://bucket_bar//_download_file?filename=foo.txt
            #   Download files at all keys in bucket_bar

        """
        s3 = self.get_connection()
        bucket = self.get_bucket(s3, url)
        parsed = urlparse.urlparse(url)
        name = parsed.netloc
        qs = parsed.query
        query_dict = urlparse.parse_qs(qs)
        path_parts = parsed.path.strip(u'/').split(u'/')
        key = path_parts[0] if len(path_parts) else ''
        func = path_parts[-1] if len(path_parts) > 1 else ''
        #   _download_file
        if func == '_download_file':
            if key == '' or key == '_all':
                for key_ in bucket.objects.all():
                    bucket.download_file(key_, key_)
            else:
                filenames = query_dict.get('filename') or []
                filename = filenames[0] if filenames else None
                resp = bucket.download_file(key, filename)
        else:
            if key == '' or key == '_all':
                resp = S3ContentList([key.get() for key in bucket.objects.all()
                                      ])
            else:
                bucket_key = s3.Object(name, key)
                resp = S3Content(bucket_key.get())
        return resp

    def get_prefixes(self):
        client = boto3.client('s3')
        paginator = client.get_paginator('list_objects')
        result = paginator.paginate(Bucket='my-bucket', Delimiter='/')
        for prefix in result.search('CommonPrefixes'):
            yield prefix.get('Prefix')

    def restore_glacier(self, url):
        s3 = self.get_connection()
        parsed = urlparse.urlparse(url)
        name = parsed.netloc
        qs = parsed.query
        bucket = self.get_bucket(s3, url)
        for obj_sum in bucket.objects.all():
            obj = s3.Object(obj_sum.bucket_name, obj_sum.key)
            if obj.storage_class == 'GLACIER':
                # Try to restore the object if the storage class is glacier and
                # the object does not have a completed or ongoing restoration
                # request.
                if obj.restore is None:
                    LOGGER.info('Submitting restoration request: {}'.format(
                        obj.key))
                    obj.restore_object()
                # Print out objects whose restoration is on-going
                elif 'ongoing-request="true"' in obj.restore:
                    LOGGER.info('Restoration in-progress: {}'.format(obj.key))
                # Print out objects whose restoration is complete
                elif 'ongoing-request="false"' in obj.restore:
                    LOGGER.info('Restoration complete: {}'.format(obj.key))

    def get_metadata(self):
        pass

    def put_metadata(self, key):
        key.put(Metadata={'meta1': 'This is my metadata value'})
        LOGGER.info(key.metadata['meta1'])

    def do_delete(self, url):
        pdb.set_trace()
        s3 = self.get_connection()
        bucket = self.get_bucket(s3, url)
        parsed = urlparse.urlparse(url)
        name = parsed.netloc
        if parsed.path:
            #   Delete the key
            key = parsed.path.strip(u'/').split(u'/')[0]
            resps = []
            for key_obj in bucket.objects.filter(Prefix=key, MaxKeys=1):
                LOGGER.info(key_obj)
                if key == key_obj.key:
                    resp_key = key_obj.delete()
                    resps.append(resp_key)
            resp = S3DictList(resps)
        else:
            #   Delete entire bucket
            resps = []
            for key_obj in bucket.objects.all():
                resp_key = key_obj.delete()
                LOGGER.info(resp_key)
                resps.append(resp_key)
            resp_bucket = bucket.delete()
            resps.append(resp_bucket)
            resp = S3DictList(resps)
        return resp

    def do_post(self, url, body):
        raise NotImplementedError("POST requests are not supported.")

    def do_put(self, url, body):
        """
        s3://bucket_bar/key_foo
            #   Put bucket_bar.key_foo
        s3://bucket_bar/key_foo/_upload_file
            #   Upload file key_foo to bucket_bar.key_foo
        s3://bucket_bar/key_foo/_upload_file?filename=foo.txt
            #   Upload file foo.txt to bucket_bar.key_foo

        """
        s3 = self.get_connection()
        parsed = urlparse.urlparse(url)
        name = parsed.netloc
        try:
            bucket = self.get_bucket(s3, url)
        except MissingBucket as exc:
            bucket = s3.create_bucket(Bucket=name)
        qs = parsed.query
        query_dict = urlparse.parse_qs(qs)
        path_parts = parsed.path.strip(u'/').split(u'/')
        key = path_parts[0] if len(path_parts) else ''
        func = path_parts[-1] if len(path_parts) > 1 else ''
        #   _upload_file
        if func == '_upload_file':
            if key == '' or key == '_all':
                raise Exception("Can only upload a single file.")
            else:
                filenames = query_dict.get('filename') or []
                filename = filenames[0] if filenames else None
                resp = bucket.upload_file(filename, key)
        else:
            #   Add data
            resp = bucket.put_object(Key=key, Body=body)
            LOGGER.info(resp)
        return resp

    def send(self, request, *args, **kwargs):
        if request.method == 'GET':
            s3_resp = self.do_get(request.url)
        elif request.method == 'DELETE':
            s3_resp = self.do_delete(request.url)
        elif request.method == 'POST':
            #   Not implemented.
            s3_resp = self.do_post(request.url, request.body)
        elif request.method == 'PUT':
            s3_resp = self.do_put(request.url, request.body)
        resp = self.build_response(request, s3_resp)
        return resp

    def close(self):
        pass


def test_s3():
    sess = requests.Session()
    sess.mount('s3', S3Adapter())

    urls = [
        {'method': 'put',
         'url': "s3://test.bar/test.foo",
         "body": "put test.foo"},
        # {'method': 'put', 'url': "s3://test.bar/test.baz", "body": "put test.baz"},
        # {'method': 'put', 'url': "s3://test.bar/test.foo/_upload_file", "params": {"filename": "foo_upload.txt"}},
        # {'method': 'get', 'url': "s3://test.bar/test.foo"},
        # {'method': 'get', 'url': "s3://test.bar/test.foo/_download_file", "params": {"filename": "foo_download.txt"}},
        {'method': 'get',
         'url': "s3://test.bar"},

        #{'method': 'get', 'url': "s3://test.bar/_all/_download_file", "params": {"filename": "foo_download.txt"}},
        #{'method': 'get', 'url': "s3://test.bar//_download_file", "params": {"filename": "foo_download.txt"}},

        # {'method': 'delete', 'url': "s3://test.bar/test.foo"}
    ]
    if not os.path.exists('foo_upload.txt'):
        with open('foo_upload.txt', 'wb') as f_out:
            f_out.write('This is a test!\n')
    for url_info in urls:
        LOGGER.info(pformat(url_info))
        if url_info['method'] == 'put':
            if url_info.get('params'):
                url_info['url'] = u'{}?{}'.format(
                    url_info['url'], urllib.urlencode(url_info['params']))
            resp = sess.put(url_info['url'], data=url_info.get('body'))
        elif url_info['method'] == 'get':
            if url_info.get('params'):
                url_info['url'] = u'{}?{}'.format(
                    url_info['url'], urllib.urlencode(url_info['params']))
            resp = sess.get(url_info['url'], params=url_info.get('params'))
        elif url_info['method'] == 'delete':
            resp = sess.delete(url_info['url'])
            # pdb.set_trace()
        LOGGER.info(resp.content)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_s3()
