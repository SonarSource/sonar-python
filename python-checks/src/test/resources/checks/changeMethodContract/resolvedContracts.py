from abc import abstractmethod as abstract
from typing import overload
from typing_extensions import overload as extension_overload

class Root:
    def run(self, value): ...

class Left(Root):
    pass

class Right(Root):
    def run(self, value, optional=None): ...

class Diamond(Left, Right):
    def run(self, value): ...  # Noncompliant {{Add missing parameters optional.}}

class UnresolvedFirst(missing.Base, Root):
    def run(self): ...  # Unknown MRO: do not select Root speculatively.

class Abstract:
    @abstract
    def run(self, value): ...

class Concrete(Abstract):
    def run(self): ...  # Noncompliant

class Decorated:
    @unknown_decorator
    def run(self, value): ...

class UnknownDecorator(Decorated):
    def run(self): ...

class UnknownOverride(Root):
    @unknown_decorator
    def run(self): ...

class DifferentDefaults:
    @overload
    def run(self, value: int = 0): ...
    @overload
    def run(self, value: str): ...

class AmbiguousContract(DifferentDefaults):
    def run(self, value): ...  # Alternative defaults disagree: no arbitrary selection.


class SameContract:
    @overload
    def run(self, value: int): ...
    @overload
    def run(self, value: str): ...

class SameContractChild(SameContract):
    def run(self): ...  # Noncompliant {{Add missing parameters value.}}


class ExtensionContract:
    @extension_overload
    def run(self, value: int): ...
    @extension_overload
    def run(self, value: str): ...
    def run(self, value): ...


class ExtensionContractChild(ExtensionContract):
    def run(self): ...  # Noncompliant {{Add missing parameters value.}}

class ShadowedAbstract:
    @custom.abstractmethod
    def run(self, value): ...

class ShadowedAbstractChild(ShadowedAbstract):
    def run(self): ...

class Replacement(Root):
    run = 42

class NonMethodOverride(Replacement):
    def run(self): ...
