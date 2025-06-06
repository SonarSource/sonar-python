import asyncio
import trio
import anyio


async def test_asyncio_taskgroup_noncompliant():
    async def worker_task():
        await asyncio.sleep(1)

    async with asyncio.TaskGroup() as tg1:  # Noncompliant {{Replace the TaskGroup with a direct function call when it only ever spawns one task.}}
        #                             ^^^
        tg1.create_task(worker_task())
    #   ^^^^^^^^^^^^^^^< {{Only task created here}}

    async def another_task():
        return 42

    async with asyncio.TaskGroup() as tg2:  # Noncompliant
        tg2.create_task(another_task())

    async with asyncio.TaskGroup() as tg:  # Noncompliant
        tg.create_task(worker_task())
        print("doing other work")

async def test_asyncio_taskgroup_compliant():
    async def worker_task():
        await asyncio.sleep(1)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(worker_task())
        tg.create_task(worker_task())

    async with asyncio.TaskGroup() as tg:
        pass

    async def task_with_tg(tg):
        await asyncio.sleep(1)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(task_with_tg(tg))

    async with asyncio.TaskGroup() as tg:
        task_with_tg(tg)

async def test_trio_nursery_noncompliant():
    async with trio.open_nursery() as nursery:  # Noncompliant {{Replace the Nursery with a direct function call when it only ever spawns one task.}}
        #                             ^^^^^^^
        nursery.start_soon(worker_task)

    async def task_with_args(x, y):
        await trio.sleep(x + y)

    async with trio.open_nursery() as nursery:  # Noncompliant
        nursery.start_soon(task_with_args, 1, 2)

    async with trio.open_nursery() as nursery:  # Noncompliant
        nursery.start(task_fn)


async def test_trio_nursery_compliant():
    async def worker_task():
        await trio.sleep(1)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(worker_task)
        nursery.start_soon(worker_task)

    async with trio.open_nursery() as nursery:
        pass

    async def task_with_nursery(nursery):
        await trio.sleep(1)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(task_with_nursery, nursery)

    async def task_fn(task_status):
        task_status.started()
        await trio.sleep(1)

    async with trio.open_nursery() as nursery:
        nursery.start(task_fn)
        nursery.start_soon(worker_task)


async def test_anyio_taskgroup_noncompliant():
    async with anyio.create_task_group() as tg:  # Noncompliant
        tg.start_soon(worker_task)

    async def task_with_args(x, y):
        await anyio.sleep(x + y)

    async with anyio.create_task_group() as tg:  # Noncompliant
        tg.start_soon(task_with_args, 1, 2)

    async def task_fn():
        await anyio.sleep(1)

    async with anyio.create_task_group() as tg:  # Noncompliant
        await tg.start(task_fn)


    async with anyio.create_task_group() as tg:
        if condition:
            tg.start_soon(anyio.sleep, 1)
        else:
            tg.start_soon(anyio.sleep, 2)


async def test_anyio_taskgroup_compliant():
    async def worker_task():
        await anyio.sleep(1)

    async with anyio.create_task_group() as tg:
        tg.start_soon(worker_task)
        tg.start_soon(worker_task)

    async with anyio.create_task_group() as tg:
        pass

    async def task_with_tg(tg):
        await anyio.sleep(1)

    async with anyio.create_task_group() as tg:
        tg.start_soon(task_with_tg, tg)


async def test_edge_cases():
    async with asyncio.TaskGroup() as outer_tg:  # Noncompliant

        async def inner_task():
            async with asyncio.TaskGroup() as inner_tg:
                inner_tg.create_task(asyncio.sleep(1))
                inner_tg.create_task(asyncio.sleep(2))

        outer_tg.create_task(inner_task())

    async with trio.open_nursery() as nursery:  # Noncompliant
        nursery.start_soon(lambda: trio.sleep(1))

    class TaskRunner:
        async def run_task(self):
            await asyncio.sleep(1)

    runner = TaskRunner()
    async with asyncio.TaskGroup() as tg:  # Noncompliant
        tg.create_task(runner.run_task())

    condition = True
    async with anyio.create_task_group() as tg:  # Noncompliant
        if condition:
            tg.start_soon(anyio.sleep, 1)

    async with anyio.create_task_group() as tg:  # Noncompliant
        try:
            tg.start_soon(anyio.sleep, 1)
        except Exception:
            pass

async def test_loops():
    async with asyncio.TaskGroup() as tg:
        for i in range(1):
            tg.create_task(asyncio.sleep(i))

    async with trio.open_nursery() as nursery:
        while True:
            nursery.start_soon(trio.sleep, 1)
            break

    async with anyio.create_task_group() as tg:
        for _ in range(5):
            tg.start_soon(anyio.sleep, 1)

    async with asyncio.TaskGroup() as tg:
        i = 0
        while i < 3:
            tg.create_task(asyncio.sleep(i))
            i += 1

    async with trio.open_nursery() as nursery:
        for i in range(2):
            for j in range(3):
                nursery.start_soon(trio.sleep, i + j)

async def test_other_uses_compliant():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(trio.sleep, 1)
        nursery.cancel_scope.deadline = trio.current_time() + 10


async def coverage():

    async with not_a_call_expr as smth:
        ...

    tg = asyncio.TaskGroup()
    async with tg as smth:
        ...

    async with open("a", "r") as smth:
        ...

    with not_async() as smth:
        ...

    async with trio.open_nursery() as (smth1, smth2):
        ...

    async with asyncio.TaskGroup() as tg:
        tg.create_task(asyncio.sleep(1))
        tg.create_task(asyncio.sleep(1))
        tg.create_task(asyncio.sleep(1))
        tg.create_task(asyncio.sleep(1))
        tg.create_task(asyncio.sleep(1))
        tg.create_task(asyncio.sleep(1))


