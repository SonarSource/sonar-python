from typing import Union
from typing import Union as u
import typing
import typing as t

def function_return() -> Union[str, int]: # Noncompliant {{Use the union type expression for this type hint}}
                        #^^^^^^^^^^^^^^^
    pass

def typing_union() -> typing.Union[int, str]: # Noncompliant
                     #^^^^^^^^^^^^^^^^^^^^^^
    pass

def from_import_alias() -> u[str, float]: # Noncompliant
    pass

def import_alias() -> t.Union[float, int]: # Noncompliant
    pass

def function_param(param: Union[str, int]): # Noncompliant
                         #^^^^^^^^^^^^^^^
    pass

def local_variable():
    variable : Union[int, str] # Noncompliant
              #^^^^^^^^^^^^^^^


class MyClass:

    instance_variable: Union[int, str] # Noncompliant
                      #^^^^^^^^^^^^^^^

    def instance_method() -> Union[int, str]: # Noncompliant
                            #^^^^^^^^^^^^^^^
        pass


def ok(param: int | str) -> int | str:
    variable : int | str
    variable = param
    return variable

def not_union_type() -> None:
    pass
