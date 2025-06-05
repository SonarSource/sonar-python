import asyncio
import trio
import anyio

# Asyncio TaskGroup tests
async def asyncio_noncompliant():
    async with asyncio.TaskGroup() as tg:
    #          ^^^^^^^^^^^^^^^^^^^> {{This is an async task group context.}}
        tg.create_task(background_task())

        if condition():
            return "result"  # Noncompliant {{Refactor the block to eliminate this "return" statement.}}
        #   ^^^^^^^^^^^^^^^

        for i in range(10):
            if i > 5:
                break  # Noncompliant {{Refactor the block to eliminate this "break" statement.}}

        while condition():
            if not other_condition():
                continue  # Noncompliant {{Refactor the block to eliminate this "continue" statement.}}

async def asyncio_compliant():
    result = None
    async with asyncio.TaskGroup() as tg:
        task = tg.create_task(background_task())

        if condition():
            result = "result"
            task.cancel()

    for i in range(10):
        if i > 5:
            break  # Compliant - outside the TaskGroup
    return result  # Compliant - outside the TaskGroup


# Trio nursery tests
async def trio_noncompliant():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(background_task)

        if condition():
            return "result"  # Noncompliant

async def trio_compliant():
    result = None
    async with trio.open_nursery() as nursery:
        nursery.start_soon(background_task)

        if condition():
            result = "result"
            nursery.cancel_scope.cancel()

    return result  # Compliant - outside the nursery

# AnyIO task group tests
async def anyio_noncompliant():
    async with anyio.create_task_group() as tg:
        tg.start_soon(background_task)

        for i in range(10):
            if i > 5:
                break  # Noncompliant

        if condition():
            return "result"  # Noncompliant

async def anyio_compliant():
    result = None
    async with anyio.create_task_group() as tg:
        tg.start_soon(background_task)

        if condition():
            result = "result"
            tg.cancel_scope.cancel()

    for i in range(10):
        if i > 5:
            break  # Compliant - outside the task group

    return result  # Compliant - outside the task group

# Nested functions test
async def nested_function_test():
    async with asyncio.TaskGroup() as tg:
        def inner_function():
            return "something"  # Compliant - inside a nested function
        
        tg.create_task(background_task())
        result = inner_function()
        
        return result  # Noncompliant

# Regular with statement test
async def regular_with_statement():
    async with some_context_manager():
        return "result"  # Compliant - not in a TaskGroup or Nursery

    async with other_expression[42]:
        return "result"  # Compliant - not async

# Non-async with statement test
def non_async_with_statement():
    with some_context_manager():
        return "result"  # Compliant - not async

async def background_task():
    await asyncio.sleep(1)
    return "done"

def condition():
    return True

def other_condition():
    return False
