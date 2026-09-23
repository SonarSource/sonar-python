import nonContextManagerBackport as contextlib
from typing import AsyncGenerator, Generator


@contextlib.contextmanager
def session() -> Generator[int, None, None]:
    yield 1


@contextlib.asynccontextmanager
async def async_session() -> AsyncGenerator[int, None]:
    yield 1


class Service:
    @contextlib.asynccontextmanager
    async def state_change(self):
        yield 1

    async def run(self):
        async with self.state_change():  # Compliant
            pass


def use():
    with session() as s:  # Compliant
        print(s)


async def use_async():
    async with async_session() as s:  # Compliant
        print(s)
