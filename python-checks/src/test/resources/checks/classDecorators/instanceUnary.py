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


def unary_operations(condition):
    direct = Direct()
    -direct
    +direct
    ~direct

    child = Child()
    -child
    +child
    ~child

    plain = Plain()
    -plain  # Noncompliant
    +plain  # Noncompliant
    ~plain  # Noncompliant

    marked = Final()
    -marked  # Noncompliant
    +marked  # Noncompliant
    ~marked  # Noncompliant

    unknown = Unknown()
    -unknown
    +unknown
    ~unknown

    mixed = Final() if condition else Child()
    -mixed
    +mixed
    ~mixed

    # Still visit the operand when its outer operation has unknown effects.
    -(direct + ("text" + 1))  # Noncompliant


-Direct()
-Child()
-Final()  # Noncompliant

# runtime_checkable on an iterator protocol does not add unary methods.
-iter([])  # Noncompliant


def missing_v2_type():
    state = Plain.__getstate__(Plain())
    -state  # Noncompliant
