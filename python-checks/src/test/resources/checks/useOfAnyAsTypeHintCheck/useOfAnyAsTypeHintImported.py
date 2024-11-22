from abc import ABCMeta

class ImportedParentWithoutMetaClass: ...

class ImportedParentWithMetaClass(metaclass=ABCMeta): ...

from typing import Callable
class MyClassWithAnnotatedMember:
    my_member: Callable[[Any],Any] # No issue on nested values of "Callable"
