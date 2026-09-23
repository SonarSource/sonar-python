from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator


@contextmanager
def create_session() -> Generator[int, None, None]:
    yield 1


@asynccontextmanager
async def create_session_async() -> AsyncGenerator[int, None]:
    yield 1


def raw_generator() -> Generator[int, None, None]:
    yield 1
