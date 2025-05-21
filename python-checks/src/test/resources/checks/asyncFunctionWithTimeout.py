import asyncio
import trio
import anyio


class AsyncioClass:
    async def method_noncompliant(self, timeout): # Noncompliant {{Remove this "timeout" parameter and use a timeout context manager instead.}}
#                                       ^^^^^^^
#   ^^^^^@-1< {{This function is async.}}
        await asyncio.sleep(1)

    async def method_compliant(self):
        await asyncio.sleep(1)

    def sync_method_with_timeout(self, timeout): # Compliant
        pass

# Test cases for asyncio
async def noncompliant_simple(timeout): # Noncompliant
    await asyncio.sleep(1)

async def noncompliant_with_typehint(timeout: int): # Noncompliant
    await asyncio.sleep(1)

async def noncompliant_with_default(timeout=5): # Noncompliant
    await asyncio.sleep(1)

async def noncompliant_keyword_only(param, *, timeout): # Noncompliant
    await asyncio.sleep(1)

async def noncompliant_positional_only(timeout, /, param): # Noncompliant
    await asyncio.sleep(1)

async def timeout_different_name(custom_timeout): # OK
    await asyncio.sleep(1)

async def noncompliant_multiple_params(arg1, timeout, arg2): # Noncompliant
    await asyncio.sleep(1)

async def noncompliant_star_args(arg1, *args, timeout): # Noncompliant
    await asyncio.sleep(1)

async def noncompliant_star_kwargs(arg1, timeout, **kwargs): # Noncompliant
    await asyncio.sleep(1)

async def compliant_no_timeout_param():
    await asyncio.sleep(1)

async def compliant_timeout_handled_by_caller():
    await asyncio.sleep(1)

async def main_compliant():
    async with asyncio.timeout(5):
        await compliant_timeout_handled_by_caller()

def sync_function_with_timeout_param(timeout): # Compliant
    pass

async def timeout_in_kwargs(**kwargs): # FN
    duration = kwargs.get("timeout", 1)
    await asyncio.sleep(duration)

# Nested functions

def outer_function():
    async def nested_function_noncompliant(timeout): # Noncompliant
        await asyncio.sleep(timeout)
    await nested_function_noncompliant(5)

async def outer_function():
    def nested_function_noncompliant(timeout): # OK
        foo()
    await something()


# Inheritance

class ClassWithUnknownParent(Unknown):
    async def method_with_timeout(self, timeout): # OK, may be inherited
        await something()

class KnownClass:
    ...

class ClassWithKnownParent(KnownClass):
    async def method_with_timeout(self, timeout): # Noncompliant
        await something()

class KnownClassWithTimeout:
    async def method_with_timeout(self, timeout): # Noncompliant
        await something()

class ClassWithKnownParent(KnownClassWithTimeout):
    async def method_with_timeout(self, timeout): # OK, inherited
        await something_else()

