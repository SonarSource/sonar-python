from contextlib import asynccontextmanager


@asynccontextmanager
async def session():
    yield 1


async def run():
    # S7515 cannot identify this decorated call as an async context manager.
    with session():  # Noncompliant {{Use "async with" for this asynchronous context manager created by "@asynccontextmanager".}}
        pass
    async with session():
        pass


class AsyncManager:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@asynccontextmanager
def factory() -> AsyncManager:
    # The annotation already makes S7515 recognize the async protocol.
    return unknown_iterator()


async def asynchronous_scope():
    with factory():  # Reported only by S7515.
        pass
    async with factory():
        pass
