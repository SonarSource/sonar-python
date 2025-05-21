# --- ASYNCIO TESTS ---
import asyncio


async def asyncio_with_input():
    name = input()  # Noncompliant
    return name


# fmt: off
async def nesting():
    async def for_assertion():
#   ^^^^^> {{This function is async.}}
        name = input()  # Noncompliant {{Wrap this call to input() with await asyncio.to_thread(input).}}
        #      ^^^^^
        return name
# fmt: on


async def asyncio_with_input_in_conditional():
    if True:
        name = input()  # Noncompliant
    return name


async def asyncio_with_proper_input():
    name = await asyncio.to_thread(input)  # Compliant
    return name


async def asyncio_with_input_in_try_except():
    try:
        name = input()  # Noncompliant
    except Exception:
        name = "Default"
    return name
