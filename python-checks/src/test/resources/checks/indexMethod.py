from numpy.lib.index_tricks import mgrid
from typing import Sequence

class Index:
    def __index__(self):
        return 0

class NotIndex:
    ...

def foo():
    my_list = ["spam", "eggs"]
    my_list["spam"]  # Noncompliant
#           ^^^^^^
    my_list[1]
    my_list[1,2] # Noncompliant
    my_list[1,] # Noncompliant
    my_list[unknown()]
    my_list[Index()]
    my_list[NotIndex()] # Noncompliant
    my_list[returns_one()]
    my_list[returns_a_string()] # FN
    my_list[1:None]
    my_list[0:1]
    my_list["spam":1]  # Noncompliant
    my_list[:"spam"]  # Noncompliant
    my_list[0:1:"spam"] # Noncompliant

    obj = {"spam": 42, "eggs": 1}
    obj["spam"] # OK

    s = slice(1, 2)
    my_list[:s.start]
    my_list[s]
    mgrid[1:2:1j]

    my_tuple = 1,2,3
    my_tuple[1]
    my_tuple["foo"] # Noncompliant

    my_custom_sequence = MyCustomSequence()
    my_custom_sequence[1]
    my_custom_sequence["foo"] # Noncompliant


def returns_one():
    return 1

def returns_a_string():
    return "spam"

class MyCustomSequence(Sequence):
    ...
