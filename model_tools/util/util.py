


import hashlib

def md5_hex(val):
    """Retrieve the MD5 hash of a value as a hex string.
    """
    m = hashlib.md5()
    m.update(val)
    return m.hexdigest()


def import_func(module, func):
    _temp = __import__(module, globals(), locals(), [func], -1)
    return getattr(_temp, func)
