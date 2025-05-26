import asyncio
import trio
import anyio

# AsyncIO tests
async def asyncio_tests():
    async def no_checkpoint():
        async with asyncio.timeout(1.0):  # Noncompliant {{Add a checkpoint inside this cancellation scope.}}
        #          ^^^^^^^^^^^^^^^^^^^^
            result = expensive_computation()

    async def with_await_checkpoint():
        async with asyncio.timeout(1.0):
            result = expensive_computation()
            await asyncio.sleep(0)

    async def with_nested_await():
        async with asyncio.timeout(1.0):
            result = await async_computation()

    async def multiple_scopes():
        async with asyncio.timeout(1.0):  # Noncompliant
            with asyncio.timeout(2.0):  # Noncompliant
                result = expensive_computation()

# Trio tests
async def trio_tests():
    async def no_checkpoint():
        async with trio.move_on_after(1.0):  # Noncompliant
            result = expensive_computation()

    async def with_explicit_checkpoint():
        async with trio.move_on_after(1.0):
            result = expensive_computation()
            await trio.lowlevel.checkpoint()

    async def with_yield():
        async with trio.move_on_after(1.0):  # Compliant: async for implicitly create checkpoints
            async for item in async_generator():
                process(item)

    async def cancel_scope_variable():
        scope = trio.move_on_after(1.0)
        async with scope:  # FN
            result = expensive_computation()

    async def with_yield_expression():
        async with trio.move_on_after(1.0):  # Compliant: yield expression creates a checkpoint
            items = [1, 2, 3]
            yield items
            result = expensive_computation()

    async def with_nested_for_loop():
        async with trio.move_on_after(1.0):  # Noncompliant
            for item in range(10):
                # Regular for loops don't create implicit checkpoints
                process(item)

    async def with_checkpoint_in_for_loop():
        async with trio.move_on_after(1.0):  # Compliant
            for item in range(10):
                process(item)
                await trio.lowlevel.checkpoint()


    async def checkpoint_not_awaited():
        async with trio.move_on_after(1.0):  # Noncompliant {{Add a checkpoint inside this cancellation scope.}}
#                  ^^^^^^^^^^^^^^^^^^^^^^^
            trio.lowlevel.checkpoint()
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^< {{This checkpoint is not awaited.}}
            result = expensive_computation()

# AnyIO tests
async def anyio_tests():
    async def no_checkpoint():
        async with anyio.move_on_after(1.0):  # Noncompliant
            result = expensive_computation()

    async def with_explicit_checkpoint():
        async with anyio.move_on_after(1.0):
            result = expensive_computation()
            await anyio.lowlevel.checkpoint()

    async def with_nested_cancel_scope():
        # Should we have 2 issues here?
        async with anyio.move_on_after(1.0): # Noncompliant
            async with anyio.CancelScope(deadline=2.0):  # Noncompliant
                result = expensive_computation()

    async def with_yield_expression():
        async with anyio.CancelScope(deadline=2.0):
            result = expensive_computation()
            yield result

    async def with_nested_for_loop():
        async with anyio.move_on_after(1.0):  # Noncompliant
            for i in range(5):
                process_data(i)


async def edge_cases():
    async def checkpoint_in_called_function():
        async def internal_checkpoint():
            await asyncio.sleep(0)
            return expensive_computation()

        async with asyncio.timeout(1.0): # Noncompliant
            # Function with checkpoint is not awaited
            result = internal_checkpoint()

    async def conditional_checkpoint():
        async with asyncio.timeout(1.0):
            if condition:
                await asyncio.sleep(0)
            result = expensive_computation()
