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
    async def async_method_without_await_trivial(self):  # Noncompliant
        return self.some_attribute

    async def async_method_without_await(self):  # Noncompliant
        do_something()
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

    async def not_implemented_error(self):
        raise NotImplementedError("This method is not implemented")

    async def not_implemented(self):
        return NotImplemented

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

async def async_comprehension():
    return [something async for something in async_iterable()]

async def sync_comprehension(): # Noncompliant
    return [something for something in async_iterable()]


# httpx.AsyncClient event hooks - coroutine callbacks are required
import httpx

async def validate_response(response: httpx.Response) -> None:  # Compliant - registered as AsyncClient hook
    validate_url(response.url)

async def validate_request(request: httpx.Request) -> None:  # Compliant - registered as AsyncClient hook
    validate_headers(request.headers)

async def both_hooks(response: httpx.Response) -> None:  # Compliant - used in both hooks
    do_something()

async_client = httpx.AsyncClient(
    event_hooks={
        "request": [validate_request, both_hooks],
        "response": [validate_response, both_hooks],
    }
)

class CustomAsyncClient(httpx.AsyncClient):
    pass

async def subclass_hook(response: httpx.Response) -> None:  # Compliant - registered as subclass of AsyncClient hook
    validate_url(response.url)

CustomAsyncClient(event_hooks={"response": [subclass_hook]})

async def sync_client_hook(response: httpx.Response) -> None:  # Noncompliant
    validate_url(response.url)

sync_client = httpx.Client(
    event_hooks={"response": [sync_client_hook]}
)

async def unrelated_hook(response) -> None:  # Noncompliant
    validate_url(response.url)

async def hook_no_event_hooks_kwarg(response) -> None:  # Noncompliant
    validate_url(response.url)

httpx.AsyncClient(timeout=hook_no_event_hooks_kwarg)  # no event_hooks kwarg

# FP: function is an event hook, but event_hooks is passed as a variable; variable indirection cannot be tracked
async def hook_via_variable(response) -> None:  # Noncompliant
    validate_url(response.url)

hooks_dict = {"response": [hook_via_variable]}
httpx.AsyncClient(event_hooks=hooks_dict)


# confluent_kafka.aio.AIOConsumer.subscribe - coroutine callbacks are required
from confluent_kafka.aio import AIOConsumer

consumer = AIOConsumer({"bootstrap.servers": "localhost:9092", "group.id": "mygroup"})

async def on_assign_handler(consumer, partitions) -> None:  # Compliant - registered as AIOConsumer.subscribe callback
    logger.info("Partitions assigned: %s", partitions)

async def on_revoke_handler(consumer, partitions) -> None:  # Compliant - registered as AIOConsumer.subscribe callback
    logger.info("Partitions revoked: %s", partitions)

async def on_lost_handler(consumer, partitions) -> None:  # Compliant - registered as AIOConsumer.subscribe callback
    logger.info("Partitions lost: %s", partitions)

await consumer.subscribe(
    ["topic"],
    on_assign=on_assign_handler,
    on_revoke=on_revoke_handler,
    on_lost=on_lost_handler,
)

async def unrelated_subscribe_callback(consumer, partitions) -> None:  # Noncompliant
    logger.info("Partitions: %s", partitions)

unknown_obj.subscribe(on_assign=unrelated_subscribe_callback)

# FP: callback is passed positionally as on_assign, but we only detect keyword arguments
async def positional_subscribe_callback(consumer, partitions) -> None:  # Noncompliant
    logger.info("Partitions: %s", partitions)

await consumer.subscribe(["topic"], positional_subscribe_callback)

# Passed via an unknown keyword argument (not on_assign/on_revoke/on_lost)
async def unknown_kwarg_subscribe_callback(consumer, partitions) -> None:  # Noncompliant
    logger.info("Partitions: %s", partitions)

await consumer.subscribe(["topic"], on_other=unknown_kwarg_subscribe_callback)
