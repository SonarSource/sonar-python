import typing
from typing import Any, override, overload


def foo(test: str, param: Any) -> str: # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
                         #^^^
    pass

def foobar() -> Any: # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
               #^^^
    my_var: Any # Noncompliant
           #^^^

    no_hint_var = 1

    pass

def multiple_hints(param: Any) -> Any: # Noncompliant 2
    pass

def multiline(param: Any, # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
                    #^^^
        param2: Any) -> Any: # Noncompliant 2
    pass

class Bar:

    my_var: Any # Noncompliant
           #^^^

    no_hint_var = "test"

    def foo(test: int, param: Any) -> str: # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
                             #^^^
        pass

    def foobar() -> Any: # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
                   #^^^
        pass

def success(param: str | int) -> None:
    pass

def success_without_hint(param):
    pass


class Parent1:
    ...



class Child1(Parent1):

    @override
    def something():
        test: Any  # Noncompliant

    @override(Parent1)
    def add_item(self, param1: Any) -> None: # Compliant
        ...

    @overload
    def add_item(self, param2: Any) -> None: # Compliant
        ...

    @override
    def add_item(self, text: str, param1: Any) -> None: # Compliant
        ...

    @typing.overload
    def add_item(self, text: str, param2: Any) -> None: # Compliant
        ...

    def over():
        def wrapper():
            ...
        return wrapper

    @over
    def add_item(self, text: str, param3: Any) -> None: # Noncompliant
        ...


class Parent2:
    def add_item(self, text: str, param1: Any) -> None: # Noncompliant
        ...

    def some_function(self, text: str) -> None:
        ...

    def text_function(self, text: Any) -> None: # Noncompliant
        ... 

    def other_function(self, text: str, other: Any) -> None: # Noncompliant
        ...

    def return_type_check(self, text) -> Any: # Noncompliant
        ...

class Child2(Parent2, object):
    def add_item(self, text: str, param1: Any) -> None: # Compliant it is an override
        ...

    def some_function(self, text: str, extra_param: Any) -> None:  # Compliant FN
        ...

    def text_function(self, text: Any, other_param) -> None: # Compliant
        ...

    # Here we consider this `other_function` as an override even if the parameters are in different order.
    def other_function(self, other: Any, text: str) -> None: # Compliant
        ...

    def return_type_check(self, text) -> Any: # Compliant
        ...

    @my_annotation[42]
    def some_annotated_method(self, text: Any):  # Noncompliant
        ...
