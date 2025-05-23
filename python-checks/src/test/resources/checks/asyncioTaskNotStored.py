import asyncio

async def noncompliant_example():
    asyncio.create_task(some_coroutine()) # Noncompliant {{Save this task in a variable to prevent premature garbage collection.}}
#   ^^^^^^^^^^^^^^^^^^^
    asyncio.ensure_future(some_coroutine())  # Noncompliant
    asyncio.create_task(some_function_call(param1, param2))  # Noncompliant

async def compliant_examples():
    # Compliant - task is stored in a variable
    task1 = asyncio.create_task(some_coroutine())
    
    # Compliant - task is stored in a variable
    task2 = asyncio.ensure_future(some_coroutine())
    
    # Compliant - tasks are stored in a collection
    tasks = [
        asyncio.create_task(some_coroutine()),
        asyncio.create_task(another_coroutine())
    ]
    
    # Compliant - task is passed to another function
    await_task(asyncio.create_task(some_coroutine()))
    
    # Compliant - used with gather
    await asyncio.gather(
        asyncio.create_task(some_coroutine()),
        asyncio.create_task(another_coroutine())
    )

async def some_coroutine():
    await asyncio.sleep(1)
    return {"result": "value"}

async def another_coroutine():
    await asyncio.sleep(0.5)
    return {"result": "other value"}

def some_function_call(p1, p2):
    return some_coroutine()

async def await_task(task):
    return await task

# Python 3.11+ TaskGroup example
async def task_group_example():
    async with asyncio.TaskGroup() as tg:
        # Compliant - TaskGroup manages the tasks
        tg.create_task(some_coroutine())
        tg.create_task(another_coroutine())
