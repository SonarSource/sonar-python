# UnionType gained class subscription in Python 3.14.
from types import UnionType
from types import UnionType as UnionAlias
import types

def union_type_subscription(left, right):
    types.UnionType[left, right]
    UnionType[int, str]
    UnionAlias[int, str]
