from typing import Any

class ClassUsingUnknownOverloadSymbols(UnknownParent):
    @overload
    def addItem(self, data: Any = ...) -> None:
        ...

    @override
    def addItem(self, data: Any = ...) -> None:
        ...

from reexport_typing_overload_override import reexported_override, reexported_overload

class ClassWithReExportsFP(UnknownParent):
    # The 2 methods are FPs : see SONARPY-1673
    @reexported_override
    def addItem(self, data: Any = ...) -> None: # Noncompliant
        ...

    @reexported_overload
    def addItem(self, data: Any = ...) -> None: # Noncompliant
        ...
