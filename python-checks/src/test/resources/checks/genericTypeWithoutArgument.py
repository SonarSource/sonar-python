import typing


def foo(param: list): # Noncompliant {{Add a type argument to this generic type.}}
              #^^^^
    pass

def foobar(param: str) -> set: # Noncompliant
                         #^^^
    pass

def with_tuple(param:str) -> tuple: # Noncompliant
                            #^^^^^
    pass

def with_var(param:str):
    my_var: dict # Noncompliant
           #^^^^
    return True

def nested(my_list: list[tuple[int, dict, str]]): # Noncompliant
                                   #^^^^
    pass

def nest_types() -> typing.List[typing.Dict]: # Noncompliant
                               #^^^^^^^^^^^
    pass

class Bar:

    def foo(param: list): # Noncompliant
                  #^^^^
        pass

    def foobar(param: str) -> set: # Noncompliant
                             #^^^
        pass

    def with_var(param:str):
        my_var: dict # Noncompliant
               #^^^^
        return True

    def tuple(param: typing.Tuple): # Noncompliant
                    #^^^^^^^^^^^^
       pass

class TypingClass:

    UserType = NewType('UserType', dict)

    user: UserType

    def foo(param: typing.List): # Noncompliant
                  #^^^^^^^^^^^
        pass

    def foobar(param: str) -> typing.Set: # Noncompliant
                             #^^^^^^^^^^
        pass

    def with_var(param: str):
        my_var: typing.Dict # Noncompliant
               #^^^^^^^^^^^

       pass

    def tuple(param: typing.Tuple): # Noncompliant
                    #^^^^^^^^^^^^
       pass

def success_list(param: list[str]):
    pass

def success_set(param: set[int]):
    pass

def success_dict(param: dict[str, int]):
    pass

def typing_list() -> typing.List[int]:
    pass

def typing_dict() -> typing.Dict[str, int]:
    pass

def typing_set() -> typing.Set[str]:
    pass

def typing_tuple() -> typing.Tuple[str, int, int]:
     pass

def typing_tuple_param(param:typing.Tuple[str, int, int]):
    pass


