from typing import Any
from abc import ABCMeta
from useOfAnyAsTypeHintImported import ImportedParentWithoutMetaClass, ImportedParentWithMetaClass

class LocalParentWithMetaClass(metaclass=ABCMeta): ...

class LocalWithMetaClassInherited(LocalParentWithMetaClass):
    def local_inherited_foo(self) -> Any: # Noncompliant
        ...

class ImportedWithoutMetaClassInherited(ImportedParentWithoutMetaClass):
    def imported_inherited_foo(self) -> Any: # Noncompliant
        ...
class ImportedWithMetaClassInherited(ImportedParentWithMetaClass):
    def imported_inherited_foo(self) -> Any: # Noncompliant
        ...
