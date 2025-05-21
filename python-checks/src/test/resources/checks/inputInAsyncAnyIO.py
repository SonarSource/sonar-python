# --- ANYIO TESTS ---
import anyio


async def anyio_with_input():
    name = input()  # Noncompliant
    return name


# fmt: off
async def nesting():
    async def for_assertion():
#   ^^^^^> {{This function is async.}}
        name = input()  # Noncompliant {{Wrap this call to input() with await anyio.to_thread.run_sync(input).}}
        #      ^^^^^
        return name
# fmt: on
async def anyio_with_proper_input():
    name = await anyio.to_thread.run_sync(input)  # Compliant
    return name


# --- ADVANCED CASES ---


# With input in a lambda - this should still be detected
async def async_with_lambda_input():
    get_input = lambda: input()  # Noncompliant
    name = get_input()
    return name


# Input as a parameter default value - should not trigger
def regular_function_with_input_default(
    prompt=input(),
):  # Compliant - not in async context
    return prompt


async def async_function_with_input_param(prompt):
    if not prompt:
        return input()  # Noncompliant
    return prompt


# Input assigned to a variable
input_func = input


async def async_with_aliased_input():
    name = input_func()  # Noncompliant
    return name


# Non-async function using input - should not trigger
def regular_function_with_input():
    return input()  # Compliant - not in async context


# Async generator with input
async def async_generator_with_input():
    yield input()  # Noncompliant


# Class with async method using input
class AsyncClass:
    async def method_with_input(self):
        return input()  # Noncompliant

    def regular_method_with_input(self):
        return input()  # Compliant - not in async context


# Async comprehensions with input
async def async_with_comprehension():
    # This is not a realistic example but tests for correct handling
    results = [input() for _ in range(1)]  # Noncompliant
    return results


# Async with input inside a non-async function - shouldn't trigger
def function_with_async_inside():
    async def inner():
        return input()  # Noncompliant

    # This is just a definition, not a call, so no issue yet
    return inner


# Multiple input calls in the same function
async def multiple_inputs():
    first = input()  # Noncompliant
    second = input()  # Noncompliant
    return first, second


# Indirectly called input
async def indirect_input():
    def get_input():
        return input()

    return get_input()  # Technically an FN


# Input in a branch that's never executed - should still be detected
async def async_with_unreachable_input():
    if False:
        name = input()  # Noncompliant
    return "Default"


# Imported input function with alias
from builtins import input as get_input


async def async_with_imported_input():
    return get_input()  # Noncompliant
