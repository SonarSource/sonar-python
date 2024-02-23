from typing import Any

class Class(UnknownParent):
    @overload
    def addItem(self, data: Any = ...) -> None:
        ...

    @override
    def addItem(self, data: Any = ...) -> None:
        ...

from reexport_typing_overload_override import reexported_override, reexported_overload

class ClassWithReExports(UnknownParent):
    @reexported_override
    def addItem(self, data: Any = ...) -> None: # Noncompliant
        ...

    @reexported_overload
    def addItem(self, data: Any = ...) -> None: # Noncompliant
        ...
