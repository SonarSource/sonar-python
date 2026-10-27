import pytest
from contextlib import contextmanager


@contextmanager
def leaked(path):
    handle = open(path)
    yield handle
    handle.close()


@contextmanager
def swallowed(path):
    handle = open(path)
    try:
        yield handle
    except BaseException:
        pass
    handle.close()
