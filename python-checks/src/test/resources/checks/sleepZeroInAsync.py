import trio
import anyio

# Test cases for the 'trio.sleep(0)' rule

async def nesting():
    #@formatter:off
    async def trio_noncompliant_cases():
#   ^^^^^> {{This function is async.}}
        await trio.sleep(0)  # Noncompliant
        #     ^^^^^^^^^^^^^ {{Use trio.lowlevel.checkpoint() instead of trio.sleep(0).}}
        await trio.sleep(0.0)  # Noncompliant

        from trio import sleep

        await sleep(0)  # Noncompliant
        await sleep(0.0)  # Noncompliant

        import trio as t

        await t.sleep(0)  # Noncompliant
        await t.sleep(0.0)  # Noncompliant

        from trio import sleep as s

        await s(0)  # Noncompliant
        await s(0.0)  # Noncompliant

        # Variations with different integer/float representations of zero
        await trio.sleep(int(0))  # FN
        await trio.sleep(float(0))  # FN
# @formatter:on

async def trio_compliant_cases():
    await trio.sleep(1)
    await trio.sleep(0.1)
    await trio.sleep(0.00001)

    await trio.lowlevel.checkpoint()

    from trio import sleep

    await sleep(1)

    import trio as t

    await t.sleep(0.5)

    from trio import sleep as s

    await s(1)

    from trio.lowlevel import checkpoint

    await checkpoint()

    from trio import lowlevel

    await lowlevel.checkpoint()

    # Using a variable that might be zero
    delay = 0
    await trio.sleep(delay)  # Noncompliant

    delay_float = 0.0
    await trio.sleep(delay_float)  # Noncompliant

    await trio.sleep()
    await trio.sleep(invalid=0)


# Test cases for the 'anyio.sleep(0)' rule


async def anyio_noncompliant_cases():
    await anyio.sleep(0)  # Noncompliant
    await anyio.sleep(0.0)  # Noncompliant

    from anyio import sleep

    await sleep(0)  # Noncompliant
    await sleep(0.0)  # Noncompliant

    import anyio as a

    await a.sleep(0)  # Noncompliant
    await a.sleep(0.0)  # Noncompliant

    from anyio import sleep as s

    await s(0)  # Noncompliant
    await s(0.0)  # Noncompliant

    # Variations with different integer/float representations of zero
    await anyio.sleep(int(0))  # FN
    await anyio.sleep(float(0))  # FN


async def anyio_compliant_cases():
    await anyio.sleep(1)
    await anyio.sleep(0.1)
    await anyio.sleep(0.0000001)

    await anyio.lowlevel.checkpoint()

    from anyio import sleep

    await sleep(1)

    import anyio as a

    await a.sleep(0.5)

    from anyio import sleep as s

    await s(1)

    from anyio.lowlevel import checkpoint

    await checkpoint()

    from anyio import lowlevel

    await lowlevel.checkpoint()

    # Using a variable that might be zero
    duration = 0
    await anyio.sleep(duration)  # Noncompliant

    duration_float = 0.0
    await anyio.sleep(duration_float)  # Noncompliant


# False Positives / Edge Cases


# --- FP Section for Trio ---
class OtherTrioLike:
    async def sleep(self, duration):
        pass


async def fp_trio_cases():
    # Not trio.sleep
    class MyTrio:
        async def sleep(self, duration):
            print(f"MyTrio sleeping for {duration}")
            pass

    mt = MyTrio()
    await mt.sleep(0)

    # Different sleep function from another module (mocked)
    class MockTime:
        def sleep(self, duration):
            print(f"MockTime sleeping for {duration}")
            pass

    mock_time = MockTime()
    mock_time.sleep(
        0
    )  # This is not an async call, but if it were, it shouldn't be flagged

    # Locally defined sleep function
    async def sleep(duration):
        print(f"Local sleep for {duration}")
        pass

    await sleep(0)

    # Class instance with a sleep method
    instance = OtherTrioLike()
    await instance.sleep(0)

    some_other_trio = trio  # Alias
    await some_other_trio.sleep(1)
    # await some_other_trio.sleep(0) # This would be noncompliant, covered above


# --- FP Section for AnyIO ---
class OtherAnyioLike:
    async def sleep(self, duration):
        pass


async def fp_anyio_cases():
    # Not anyio.sleep
    class MyAnyIO:
        async def sleep(self, duration):
            print(f"MyAnyIO sleeping for {duration}")
            pass

    ma = MyAnyIO()
    await ma.sleep(0)

    # Different sleep function from another module (mocked)
    class MockAsyncLib:
        async def sleep(self, duration):
            print(f"MockAsyncLib sleeping for {duration}")
            pass

    mock_async_lib = MockAsyncLib()
    await mock_async_lib.sleep(0)

    # Locally defined sleep function
    async def sleep(duration):  # This sleep shadows anyio.sleep if imported directly
        print(f"Local sleep for {duration}")
        pass

    await sleep(0)

    # Class instance with a sleep method
    instance = OtherAnyioLike()
    await instance.sleep(0)

    some_other_anyio = anyio  # Alias
    await some_other_anyio.sleep(1)
    # await some_other_anyio.sleep(0) # This would be noncompliant, covered above


def not_async():
    await trio.sleep(0)
    await anyio.sleep(0)
