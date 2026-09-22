from numpy.lib.index_tricks import mgrid
from typing import Any, Mapping, Sequence

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

def mixed_union_index(index: int | str):
    [1][index] # Noncompliant
       #^^^^^

def mixed_valid_union_index(index: int | slice):
    [1][index]

def dictionary_key_is_valid(data: dict[str, str]):
    data["id"]

def any_value_can_be_a_dictionary(data: Any):
    data["id"]

class Item:
    def data(self, index, role) -> dict[str, str]:
        return {"id": "1", "text": "example"}

def update_item(item: Item, nd: dict[str, str]):
    r = item.data(0, 0)
    if r["id"] == nd["id"]:
        r["text"] = nd["text"]

class MappingSequence(Mapping, Sequence):
    ...

def mapping_keys_are_valid(mapping: MappingSequence):
    mapping["id"]

class MyCustomSequence(Sequence):
    ...

def unknown_type(data, data2: UnknownType):
    data["id"] # OK
    data2["id"] # OK

def tuple_unpacking_does_not_type_extracted_value(precomputed_cholesky):
    params, = tuple(precomputed_cholesky.values())
    params["chol_tril"]
