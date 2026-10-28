from contextlib import contextmanager as cm, asynccontextmanager as acm
from typing import Generator, AsyncGenerator


@cm
def session() -> Generator[int, None, None]:
    yield 1


@acm
async def async_session() -> AsyncGenerator[int, None]:
    yield 1


def use():
    with session() as value:
        print(value)


async def use_async():
    async with async_session() as value:
        print(value)
