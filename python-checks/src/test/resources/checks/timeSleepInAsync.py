import time
import asyncio
import trio
import anyio

# Asyncio

def asyncio_tests():
    async def noncompliant_direct_time_sleep_asyncio():
#   ^^^^^> {{This function is async.}}
        print("Asyncio: Noncompliant direct time.sleep")
        time.sleep(1) # Noncompliant {{Replace this call to time.sleep() with an asynchronous sleep function.}}
#       ^^^^^^^^^^

    async def compliant_asyncio_sleep():
        print("Asyncio: Compliant asyncio.sleep")
        await asyncio.sleep(1)

    async def noncompliant_aliased_time_sleep_asyncio():
        import time as t
        print("Asyncio: Noncompliant aliased time.sleep")
        t.sleep(1) # Noncompliant

    async def noncompliant_from_import_sleep_asyncio():
        from time import sleep
        print("Asyncio: Noncompliant from_import sleep")
        sleep(1) # Noncompliant

    def sync_function_using_time_sleep_asyncio():
        time.sleep(0.1)

    async def compliant_calling_sync_function_with_time_sleep_asyncio():
        print("Asyncio: Compliant call to sync function using time.sleep")
        sync_function_using_time_sleep_asyncio() # time.sleep is not directly in async func

    async def compliant_nested_sync_function_with_time_sleep_asyncio():
        print("Asyncio: Compliant nested sync function with time.sleep")
        def inner_sync_function():
            time.sleep(0.1) # time.sleep is in a sync function, not the outer async function
        inner_sync_function()


# Trio

def trio_test():
    async def compliant_trio_sleep():
        print("Trio: Compliant trio.sleep")
        await trio.sleep(1)

# --- Section: AnyIO ---

def anyio_tests():
    async def compliant_anyio_sleep():
        print("AnyIO: Compliant anyio.sleep")
        await anyio.sleep(1)

# --- Section: General Synchronous Cases (Compliant) ---

def general_sync_compliant_cases():
    def sync_function_with_time_sleep():
        print("General: Sync function with time.sleep")
        time.sleep(1) # Compliant

    sync_gen = (time.sleep(0.01) for _ in range(2))
    for _ in sync_gen:
        pass

# Edge cases

def edge_cases():
    async def noncompliant_lambda_with_time_sleep_in_async():
        sleeper = lambda x: time.sleep(x) # Noncompliant
        sleeper(0.1)

    async def noncompliant_lambda_assigned_and_returned():
        # Edge case: sleep is not directly in the async function
        sleeper = lambda x: time.sleep(x) # Noncompliant
        return sleeper

    # Shadowing 'time' module
    async def compliant_shadowed_time_module():
        class time: # This shadows the global time module
            @staticmethod
            def sleep(duration):
                print(f"Custom time.sleep called with {duration}")

        time.sleep(1) # OK

    # FN: Accessing time.sleep indirectly
    import time as global_time_module
    class TimeWrapper:
        def __init__(self, tm):
            self.timer = tm

        def do_sleep(self, duration):
            self.timer.sleep(duration)

    async def noncompliant_indirect_time_sleep_via_wrapper_attribute():
        wrapper = TimeWrapper(global_time_module)
        wrapper.do_sleep(1) # FN
