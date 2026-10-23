# UnionType gained class subscription in Python 3.14.
# An instance __getitem__ in its stub must not enable class subscription on older targets.
from types import UnionType
from types import UnionType as UnionAlias
import types

def union_type_subscription(left, right):
    types.UnionType[left, right]
    UnionType[int, str]
    UnionAlias[int, str]
