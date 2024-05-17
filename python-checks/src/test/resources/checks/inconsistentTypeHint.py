import re
from typing import Any, Optional, SupportsFloat, List, Iterable, Generator, Set, Union, Type, TypedDict, Sequence


def assigned_directly():
    n = None
    foo: int = n  # Noncompliant {{Replace the type hint "int" with "Optional[int]" or don't assign "None" to "foo"}}
    #    ^^^>  ^
    opt: Optional[int] = n # Ok

    t = "hello"
    my_int_nok: int = t  # Noncompliant  {{Assign to "my_int_nok" a value of type "int" instead of "str" or update its type hint.}}
    #           ^^^>  ^
    i = 42
    my_str_ok: str = i  # Noncompliant
    my_int_ok: int = i  # OK
    my_str_ok: str = t  # OK
    f: float = i # OK
    o: object = t # OK
    b = object()
    g: int = b # Noncompliant

def return_union() -> Union[str, float]:
    pass

def assigned_to_union(cond, MyType, unknown_func):
    u = return_union()
    my_int_nok: int = u  # Noncompliant  {{Assign to "my_int_nok" a value of type "int" instead of "Union[str, float]" or update its type hint.}}
    t = "hello"
    n = 42.5
    if cond:
        x = t
    else:
        x = n
    my_int_nok: int = x  # Noncompliant  {{Assign to "my_int_nok" a value of type "int" or update its type hint.}}
    union_order: float | str = x # OK
    union_order2:  str | float = x # OK
    union_any: float | Any = x # OK
    union_unknown: float | MyType = x # OK
    mT = MyType()
    unknown_type: float | str = mT # OK 
    f_call = unknown_func
    f: str | float | A = f_call # OK

def assigned_later(param: bool):
    text = "hello"
    n = 42
    a: int
    a = text  # FN
    b: int
    if param:
        b = n
    else:
        b = text  # FN

    c: int
    c = 1 if param else n  # OK
    d: int
    d = 1 if param else text  # FN


class A:
    def method():
        ...

class B(A):
    def additional_method():
        ...

class C(B):
    def other():
        ...

class DuckTypeOfA:
    def method():
        ...
    
from module1 import SomeType

class D(SomeType):
    ...

def custom_classes():
    a = A()
    b = B()
    c = C()
    d = D()

    my_a_ok: A = a  # OK
    my_a_ok2: A = b  # OK
    my_a_nok: A = A  # Noncompliant
    my_b_nok: B = a  # Noncompliant {{Assign to "my_b_nok" a value of type "B" instead of "A" or update its type hint.}}
    #         ^>  ^
    my_b_ok: B = b
    my_c_ok: A = c
    my_c_nok: C = a # Noncompliant

    my_d_ok: D = d
    my_d_ok: A = d
    my_d_ok: B = d
    my_d_ok: C = d

    duck_type = DuckTypeOfA()
    duck_type_ok: A = duck_type # OK

def get_generator():
    yield 1


def type_aliases():
    """We should avoid raising FPs on type aliases"""
    f = 42
    l = []
    g = get_generator()
    my_float: SupportsFloat = f  # OK
    my_iterable: Iterable = l  # OK
    my_generator: Generator = g  # OK

def collections():
    l = {}
    my_list: list = l  # Noncompliant

    li = [1, 2, 3]
    # Was an FN 
    my_str_list_nok: list[str] = li   # Noncompliant

    ls = ["a", "b", "c"]
    my_str_list_ok: list[str] =  ls # OK

    d = {}
    my_set_nok: set = d # Noncompliant {{Assign to "my_set_nok" a value of type "set" instead of "dict" or update its type hint.}}

    s = set
    my_set_nok2: set = s  # Noncompliant  {{Assign to "my_set_nok2" a value of type "set" instead of "type" or update its type hint.}}

    real_set = set()
    my_set_ok: Set = real_set  # OK

def generics():
    la = [A(),A()]
    lb = [B(),B()]
    lc = [C(),C()]
    v: list[B] = la # Noncompliant
    b1: list[B] = lb # OK
    b2: list[A] = lb # Noncompliant
    b3: Sequence[A] = lb # OK
    b4: Sequence[A] = lc # OK


def unresolvable(t):
    a: AType = t # OK
    b: AType = t() # OK
    
def union_any(anything):
    a = anything  # Ok
    b: int
    if param:
        b = n
    else:
        b = text  # FN

    
    

def function_params():
    def overwritten_param(param: int):
        param = "hello"  # Out of scope (S1226)

    def used_param(param: int):
        print(param)
        s = "hello"
        param = s  # FN
        print(param)


class ClassAttributes:
    s = "hello"
    my_attr: str = s  # OK
    i = 42
    my_attr_2: str = i  # Noncompliant

    my_attr_3: str

    def __init__(self):
        self.my_attr_3 = i  # Noncompliant
        hello = "hello"
        self.my_attr_4: int = hello  # Noncompliant {{Assign to this expression a value of type "int" instead of "str" or update its type hint.}}


class Meta(type):
    ...


class MyClassWithMeta(metaclass=Meta):
    ...


def metaclasses():
    s = set
    my_var: Meta = s  # Accepted FN
    my_other_var: Meta = MyClassWithMeta  # OK
    my_other_var: MyClassWithMeta = MyClassWithMeta  # Noncompliant {{Assign to "my_other_var" a value of type "MyClassWithMeta" instead of "type" or update its type hint.}}
    another_var: Type = MyClassWithMeta
    another_var: Type = set

    def a_function():
        ...

    another_var: Type = a_function  # Accepted FN
    another_var: Type = unknown_symbol


class MyCustomDict(TypedDict):
    user_ids: Set[int]
    message_ids: Set[int]


def my_dict() -> MyCustomDict:
    users = {1, 2, 3}
    messages = {1, 2, 3}
    my_dict: MyCustomDict = dict(user_ids=users, message_ids=messages)  # OK
    my_dict: MyCustomDict = {"user_ids": users, "message_ids":messages}  # OK

    return my_dict
