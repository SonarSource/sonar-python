# Test cases for async context managers

class AsyncContextManager:
#     ^^^^^^^^^^^^^^^^^^^> {{This context manager implements the async context manager protocol.}}
    async def __aenter__(self):
        print("Entering async context")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Exiting async context")
        return False

class RegularContextManager:
    def __enter__(self):
        print("Entering regular context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting regular context")
        return False

class DualContextManager:
    # Implements both sync and async protocols
    def __enter__(self):
        print("Entering regular context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting regular context")
        return False

    async def __aenter__(self):
        print("Entering async context")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Exiting async context")
        return False

async def async_function_with_async_context_manager():
#^[sc=1;ec=5]> {{This function is async.}}
    with AsyncContextManager() as acm:  # Noncompliant {{Use "async with" instead of "with" for this asynchronous context manager.}}
#   ^^^^
        print("Inside async context")

async def async_function_with_dual_context_manager():
    with DualContextManager() as dcm:  # Noncompliant
        print("Inside context")

# Compliant examples
async def async_function_with_async_with():
    async with AsyncContextManager() as acm:  # Compliant
        print("Inside async context")

async def async_function_with_regular_context_manager():
    with RegularContextManager() as rcm:  # Compliant - no async protocol
        print("Inside regular context")

async def async_function_with_dual_context_manager_compliant():
    async with DualContextManager() as dcm:  # Compliant
        print("Inside async context")

# Regular function - should be compliant regardless
def regular_function():
    with AsyncContextManager() as acm:  # Compliant - not in async function
        print("Inside async context")
    
    with DualContextManager() as dcm:  # Compliant - not in async function
        print("Inside context")

# Nested contexts
async def nested_contexts():
    with RegularContextManager() as outer:  # Compliant - no async protocol
        with AsyncContextManager() as inner:  # Noncompliant
            print("Nested contexts")

# Class method examples
class MyClass:
    async def async_method(self):
        with AsyncContextManager() as acm:  # Noncompliant
            print("Inside async method")
        
        async with AsyncContextManager() as acm:  # Compliant
            print("Inside async method with async with")
    
    def regular_method(self):
        with AsyncContextManager() as acm:  # Compliant - not in async method
            print("Inside regular method")
