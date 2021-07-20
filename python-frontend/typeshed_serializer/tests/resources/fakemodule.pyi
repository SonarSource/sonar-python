import sys
from typing import overload

if sys.version_info >= (3, 8):
    class SomeClassUnique38:
        def some_method(self): ...


    class SomeClassMultipleDefinition(Exception):
        ...
else:
    class SomeClassUnique36:
        def some_method(self): ...


    class SomeClassMultipleDefinition(str):
        ...


class CommonClass:
    def common_method(self):
        ...

    if sys.version_info >= (3, 8):
        def method_unique_38(self):
            ...

        def common_method_multiple_definition(self, param1: int, param2: str):
            ...
    else:
        def method_unique_36(self):
            ...

        def common_method_multiple_definition(self, param1: str, param2: int, param3: bool):
            ...


def common_function():
    ...

if sys.version_info >= (3, 8):
    def function_unique_38():
        ...

    def common_function_multiple_defs(param: int):
        ...
else:
    def function_unique_36():
        ...

    def common_function_multiple_defs(param: str):
        ...

@overload
def common_overloaded_function():
    ...

@overload
def common_overloaded_function():
    ...

if sys.version_info >= (3, 8):
    @overload
    def overloaded_function_38():
        ...


    @overload
    def overloaded_function_38():
        ...

    @overload
    def overloaded_function_multiple_defs(param: float):
        ...

    @overload
    def overloaded_function_multiple_defs(param: str):
        ...
else:
    @overload
    def overloaded_function_36():
        ...

    @overload
    def overloaded_function_36():
        ...

    @overload
    def overloaded_function_multiple_defs(param: int):
        ...

    @overload
    def overloaded_function_multiple_defs(param: str):
        ...
