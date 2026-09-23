# A vendored / back-ported copy of contextlib (mirrors vaex's "from .vendor import contextlib").
def contextmanager(func):
    return func


def asynccontextmanager(func):
    return func
