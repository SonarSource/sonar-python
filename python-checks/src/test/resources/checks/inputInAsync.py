import asyncio, trio, anyio as aio  # noqa: E401
import asyncio.tasks
import something_else


async def foo():
    input()  # Noncompliant {{Wrap this call to input() with the appropriate function from the asynchronous library.}}


def prevent_optimization():
    asyncio.run(foo)
    trio.run(foo())
    aio.run(foo())
    something_else.run(foo())
