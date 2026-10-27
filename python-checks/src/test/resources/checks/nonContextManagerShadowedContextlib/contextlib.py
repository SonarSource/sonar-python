class SyncManager:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class AsyncManager:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def asynccontextmanager(func):
    def wrapper(*args, **kwargs):
        return SyncManager()
    return wrapper


def contextmanager(func):
    def wrapper(*args, **kwargs):
        return AsyncManager()
    return wrapper
