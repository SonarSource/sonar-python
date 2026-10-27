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
from sqlalchemy.types import TypeDecorator


class NetworkType(TypeDecorator):
    # Known FP: this callback's two-argument contract is discovered by computed getattr
    # in unusedFunctionParameterImport.py, rather than declared by TypeDecorator.
    def alembic_render_type(self, context, toplevel_code):  # Noncompliant
        context.imports.add("from example import NetworkType")
        return "NetworkType()"
