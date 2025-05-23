import asyncio
import os
import trio
import anyio
from functools import partial

async def asyncio_with_os_waitpid():
#^[sc=1;ec=5]> {{This function is async.}}
    pid, status = os.waitpid(123, 0)  # Noncompliant {{Use a thread executor to wrap blocking OS calls in this async function.}}
#                 ^^^^^^^^^^

async def asyncio_with_os_wait():
    pid, status = os.wait()  # Noncompliant


# Framework compliant solutions
async def asyncio_compliant():
    loop = asyncio.get_running_loop()
    pid, status = await loop.run_in_executor(None, partial(os.waitpid, 123, 0))
    pid, status = await loop.run_in_executor(None, os.wait)
    res = await loop.run_in_executor(None, partial(os.waitid, os.P_PID, 123, os.WEXITED))

    def nested_sync_function():
        return os.waitpid(123, 0)

async def trio_compliant():
    pid, status = await trio.to_thread.run_sync(os.waitpid, 123, 0)
    pid, status = await trio.to_thread.run_sync(os.wait)

async def anyio_compliant():
    pid, status = await anyio.to_thread.run_sync(os.waitpid, 123, 0)
    pid, status = await anyio.to_thread.run_sync(os.wait)
