class ImportedParent:
    def using_child_method(self):
        return self.method_defined_in_child_class_only(1,2)

class DuplicatedParent:
    ...

class DuplicatedParent:
    ...

class ParentWithDuplicatedParent(DuplicatedParent):
    ...

from typing import Callable
class MyClassWithAnnotatedMember:
    my_member: Callable[[str, int],str]
