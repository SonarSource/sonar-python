import trio
import anyio
import trio as t
import anyio as a
import something_else

async def long_sleep_trio():
    await trio.sleep(86401)  # Noncompliant

async def long_sleep_trio():
    await trio.sleep()  # OK

async def long_sleep_anyio():
    await anyio.sleep(86401)  # Noncompliant {{Replace this call with "anyio.sleep_forever()" as the sleep duration exceeds 24 hours.}}

async def long_sleep_trio_arithmetic():
    await trio.sleep(86400 * 2)  # FN

async def long_sleep_with_arithmetic_2():
    days = 30
    await trio.sleep(86400 * days)  # FN

async def long_sleep_with_alias():
    await t.sleep(86401)  # Noncompliant {{Replace this call with "trio.sleep_forever()" as the sleep duration exceeds 24 hours.}}
    await a.sleep(86401)  # Noncompliant {{Replace this call with "anyio.sleep_forever()" as the sleep duration exceeds 24 hours.}}

# Compliant cases - duration <= 24 hours
async def normal_sleep_trio():
    await trio.sleep(3600)  # Compliant - one hour

async def normal_sleep_anyio():
    await anyio.sleep(86400)  # Compliant - exactly 24 hours

def not_async_function():
    # Not in async context, so not reported
    trio.sleep(86400 * 10)  # Compliant - not in async context

# Edge cases
async def with_complex_expressions():
    x = get_duration()
    await trio.sleep(x)  # Compliant - can't determine value statically

    # Complex expressions that can be evaluated
    await anyio.sleep(24 * 60 * 60 * 2)  # FN
    await trio.sleep(86400 / 0.5)  # FN

async def sleep_forever():
    # Already using the recommended alternatives
    await trio.sleep_forever()  # Compliant
    await anyio.sleep_forever()  # Compliant
