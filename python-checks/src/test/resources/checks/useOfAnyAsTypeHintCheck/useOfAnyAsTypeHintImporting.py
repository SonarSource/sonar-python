from typing import Any
from abc import ABCMeta
from useOfAnyAsTypeHintImported import ImportedParentWithoutMetaClass, ImportedParentWithMetaClass, MyClassWithAnnotatedMember

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

class MyChild(MyClassWithAnnotatedMember):
    def my_member(self, param: Any) -> Any: # OK, defined in parent
        ...
