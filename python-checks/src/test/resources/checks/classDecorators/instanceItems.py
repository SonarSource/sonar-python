from typing import final
from importedDecorators import add_protocol, ImportedBase


@add_protocol
class Direct:
    pass


class Child(ImportedBase):
    pass


class Plain:
    pass


@final
class Final:
    pass


@unknown_factory()
class Unknown:
    pass


def item_operations(condition):
    direct = Direct()
    direct[0]
    direct[0] = 42
    del direct[0]

    child = Child()
    child[0]
    child[0] = 42
    del child[0]

    plain = Plain()
    plain[0]  # Noncompliant
    plain[0] = 42  # Noncompliant
    del plain[0]  # Noncompliant

    marked = Final()
    marked[0]  # Noncompliant
    marked[0] = 42  # Noncompliant
    del marked[0]  # Noncompliant

    unknown = Unknown()
    unknown[0]
    unknown[0] = 42
    del unknown[0]

    mixed = Final() if condition else Child()
    mixed[0]
    mixed[0] = 42
    del mixed[0]

    # An unknown outer subscription must not hide an invalid inner operation.
    direct[42[0]]  # Noncompliant


# Constructor expressions take the same path as instance variables.
Direct()[0]
Child()[0]
Final()[0]  # Noncompliant
