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
