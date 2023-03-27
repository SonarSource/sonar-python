from typing import Any

def foo(test: str, param: Any) -> str: # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
                         #^^^
    pass

def foobar() -> Any: # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
               #^^^
    my_var: Any # Noncompliant
           #^^^

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

    def foo(test: int, param: Any) -> str: # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
                             #^^^
        pass

    def foobar() -> Any: # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
                   #^^^
        pass

def success(param: str | int) -> None:
    pass
