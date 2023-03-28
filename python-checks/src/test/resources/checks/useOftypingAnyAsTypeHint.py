import typing

def foo(test: str, param: typing.Any) -> str: # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
                         #^^^^^^^^^^
    pass

def foobar() -> typing.Any: # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
               #^^^^^^^^^^
    my_var: typing.Any # Noncompliant
           #^^^^^^^^^^
    pass

def multiple_hints(param: typing.Any) -> typing.Any: # Noncompliant 2
    pass

def multiline(param: typing.Any, # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
                    #^^^^^^^^^^
        param2: typing.Any) -> typing.Any: # Noncompliant 2
    pass

class Bar:

    my_var: typing.Any # Noncompliant
           #^^^^^^^^^^
    correct_var: str

    no_hint_var = "test"

    def foo(test: int, param: typing.Any) -> str: # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
                             #^^^^^^^^^^
        pass

    def foobar() -> typing.Any: # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
                   #^^^^^^^^^^
        pass

def success(param: str | int) -> None:
    pass

def success_without_hint(param):
    pass
