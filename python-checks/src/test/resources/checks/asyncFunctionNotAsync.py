import asyncio

async def noncompliant_function():  # Noncompliant {{Use asynchronous features in this function or remove the `async` keyword.}}
#         ^^^^^^^^^^^^^^^^^^^^^
    print("This function does nothing asynchronous")
#^[sc=1;ec=5]@-2<

async def return_from_sync_call():  # Noncompliant
    def inner_func():
        return "result"
    return inner_func()

async def loop_function():  # Noncompliant
    for i in range(10):
        print(i)

async def with_await():  # Compliant
    result = await some_coroutine()
    return result

async def with_async_for():  # Compliant
    async for item in async_iterable:
        print(item)

async def with_async_with():  # Compliant
    async with async_context_manager:
        print("Inside async context")

async def non_async_with():  # Noncompliant
    with context_manager:
        print("Inside async context")

async def with_create_task():  # Compliant
    task = asyncio.create_task(some_coroutine())
    await task

async def empty_function():
    pass

async def empty_function_2():
    ...

async def empty_function_3():
    """empty for now"""
    ...

async def nested_async():  # Compliant
    await some_coroutine()

    async def inner():  # Noncompliant
        print("inner function")

    return await another_coroutine()

async def await_in_comprehension():  # Compliant
    results = [await coro() for coro in coroutines]
    return results

async def nested_noncompliant():  # Noncompliant
    def inner():
        async def deeply_nested():
            return await some_coroutine()
        return deeply_nested

    return inner()()

async def sleep_without_await():  # Noncompliant
    asyncio.sleep(1)  # Missing await

# Simple async generator
async def my_async_generator():  # Compliant
    yield something()

# Async generator with yield expression
async def async_generator_with_expression():  # Compliant
    x = (yield 42)
    return x

class AsyncClass:
    async def async_method_without_await(self):  # Noncompliant
        return self.some_attribute

    async def async_method_with_await(self):  # Compliant
        return await self.some_coroutine()

    @classmethod
    async def async_classmethod_without_await(cls):  # Noncompliant
        return cls.some_value

    async def async_method_with_inner_function(self):
        async def inner_function():
            return await self.some_coroutine()
        return await inner_function()

    @abstractmethod
    async def abstract_async_method(self):  # Compliant
        raise NotImplementedError("This is an abstract method")

    @abc.abstractmethod
    async def abstract_async_method_2(self):  # Compliant
        raise NotImplementedError("This is an abstract method")

    @abc.other
    async def other_decorator_1(self):  # Noncompliant
        raise NotImplementedError("...")

    @unknown()
    async def other_decorator_1(self):  # Noncompliant
        raise NotImplementedError("...")
