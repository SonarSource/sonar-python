def sync_func():
    pass

async def async_func():
    pass

async def async_func_caller():
    await async_func

class MyClass:

    def sync_meth(self):
        pass

    async def async_meth(self):
        pass

    async = 2

def sync_function_2(fn):
    @wraps(fn)
    async def wrapper(*args):
        pass

async def foo(x):  
    async with x.bar() as baz:
        pass
    async for var in x:
        pass
