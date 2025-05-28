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
    async def async_classmethod_without_await(cls):  # Avoid FPs with decorators
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
    async def other_decorator_1(self):  # Avoid FPs with decorators
        raise NotImplementedError("...")

    @unknown()
    async def other_decorator_1(self):  # Avoid FPs with decorators
        raise NotImplementedError("...")

# Async protocol methods - should be compliant even without await
class AsyncContextManager:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __unknown_dunder__(self):
        # Avoid risk of FPs
        pass

class AsyncIterator:
    async def __aiter__(self):
        return self

    async def __anext__(self):
        if self.should_stop():
            raise StopAsyncIteration
        return self.value

class AsyncResource:
    async def __aclose__(self):
        print("Releasing resources")
        
class AsyncAwaitableObject:
    async def __await__(self):
        yield "something"

    async def regular_method_without_await(self):  # Noncompliant
        print("This is not a protocol method")

# FastAPI route examples
from fastapi import FastAPI, APIRouter

app = FastAPI()
router = APIRouter()

@app.get("/items/{item_id}")
async def read_item(item_id: int):  # Compliant - FastAPI route
    return {"item_id": item_id}

@app.post("/users/")
async def create_user(user_data: dict):  # Compliant - FastAPI route
    # No await, but this is still valid for FastAPI routes
    return {"user_id": 123, "data": user_data}

@router.put("/items/{item_id}")
async def update_item(item_id: int, item: dict):  # Compliant - FastAPI route via router
    return {"item_id": item_id, "item": item}

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):  # Compliant - FastAPI route
    # No await, but this is still valid for FastAPI routes
    return {"deleted": True}


class MyClass:
    async def my_method(self):
        await something()

class MyOtherClass(MyClass):
    async def my_method(self):
        # No issue on overriding methods
        do_something()
