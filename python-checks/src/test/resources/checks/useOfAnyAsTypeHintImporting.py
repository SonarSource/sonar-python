from typing import Any
from abc import ABCMeta
from useOfAnyAsTypeHintImported import ImportedParentWithMetaClass

class LocalParent: ...

class LocalParentWithMetaClass(metaclass=ABCMeta): ...

class LocalWithMetaClassInherited(LocalParentWithMetaClass):
    def local_inherited_foo(self) -> Any: # Noncompliant
        ...

class ImportedWithMetaClassInherited(ImportedParentWithMetaClass):
    def imported_inherited_foo(self) -> Any: # FN SONARPY-2331
        ...
