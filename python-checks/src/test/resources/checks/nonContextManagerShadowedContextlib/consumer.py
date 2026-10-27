from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator


@asynccontextmanager
async def session() -> AsyncGenerator[int, None]:
    yield 1


with session():  # Compliant: the project-local decorator produces a sync manager
    pass


@contextmanager
def async_session() -> Generator[int, None, None]:
    yield 1


async def run():
    async with async_session():  # Compliant: the project-local decorator produces an async manager
        pass
