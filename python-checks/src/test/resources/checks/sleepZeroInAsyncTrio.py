import trio

# Test cases for the 'trio.sleep(0)' rule

async def nesting():
    #@formatter:off
    async def trio_noncompliant_cases():
#   ^^^^^> {{This function is async.}}
        await trio.sleep(0)  # Noncompliant {{Use trio.lowlevel.checkpoint() instead of trio.sleep.}}
        #     ^^^^^^^^^^^^^
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

import unrelated_module
import another.unrelated_module
unrelated_module
another.unrelated_module
