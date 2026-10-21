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
    [1][index]  # A union with a valid alternative is not definitely invalid.

def invalid_union_index(index: str | float):
    [1][index]  # Noncompliant
       #^^^^^
    [1][index:index:index]  # Noncompliant 3

def custom_or_invalid_union_index(index: Index | str):
    [1][index]

def unknown_union_index(index: Any | str):
    [1][index]

def nullable_descriptor_slice_bound():
    import re
    items = [1, 2, 3]
    index = re.search("missing", "text")
    if index is None:
        items[index:index:index]

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


def conservative_nullable_index(index: str | None):
    [1][index]  # FN: None indices remain permitted for sequences with custom indexing.
    [1][index:index:index]  # None remains a possible valid slice bound.

def conservative_slice_bound(bound: str | slice):
    [1][:bound]  # FN: Slice bounds remain permitted for sequences with custom slicing.
    [1][bound:bound:bound]
    [1][bound]  # slice remains a possible valid subscription.

def definite_none_index_and_slice_bound():
    [1][None]  # FN: The rule does not distinguish built-in indexing from custom indexing.
    [1][None:None:None]
    bound = slice(1)
    [1][:bound]  # FN: The same conservative policy applies to slice bounds.
    [1][bound]

def possibly_valid_slice_bounds(valid: int | slice, custom: Index | str, unknown: Any | slice):
    [1][valid:valid:valid]
    [1][custom:custom:custom]
    [1][unknown:unknown:unknown]


class CustomIndexList(list):
    def __getitem__(self, index):
        if index is None or isinstance(index, slice):
            return self
        return super().__getitem__(index)


def custom_indexing_accepts_none_and_slice_bounds():
    items = CustomIndexList([1, 2, 3])
    assert items[None] is items
    bound = slice(1)
    assert items[bound:bound:bound] is items
