from dataclasses import dataclass, field, InitVar
from datetime import datetime
from typing import ClassVar, Optional


# =======================
# Issues
# =======================

@dataclass
class SingleUnannotatedAttribute:
    timeout = 30  # Noncompliant {{Add a type annotation to this dataclass attribute.}}
#   ^^^^^^^^^^^^


@dataclass
class MultipleUnannotatedAttributes:
    timeout = 30   # Noncompliant {{Add a type annotation to this dataclass attribute.}}
#   ^^^^^^^^^^^^
    retries = 3    # Noncompliant {{Add a type annotation to this dataclass attribute.}}
#   ^^^^^^^^^^^


@dataclass
class MixedAnnotatedAndUnannotated:
    name: str = "default"
    timeout = 30  # Noncompliant
#   ^^^^^^^^^^^^


@dataclass(frozen=True)
class FrozenDataclassUnannotated:
    timeout = 30  # Noncompliant


@dataclass
class MultipleViolations:
    unannotated = 42                       # Noncompliant
    other = "value"                        # Noncompliant


import dataclasses

@dataclasses.dataclass
class QualifiedDecoratorNoncompliant:
    unannotated = 42  # Noncompliant


@dataclass
class ChildDataclassWithViolation:
    unannotated = 99  # Noncompliant
    extra: int = 0


def outer_function():
    # Dataclass identity is also resolved for classes nested in a function.
    @dataclass
    class InnerDataclass:
        value = 10  # Noncompliant
        name: str = ""


# =======================
# Compliant
# =======================

class RegularClassWithUnannotatedAttrs:
    timeout = 30
    retries = 3


@dataclass
class ImmutableScalarDefaults:
    count: int = 0
    name: str = ""
    flag: bool = False
    ratio: float = 1.0


@dataclass
class NoneDefault:
    opt: Optional[str] = None


@dataclass
class MutableDefaultsAreNotFlagged:
    members: list = []
    config: dict = {}
    tags: set = {1, 2}


@dataclass
class CallDefaultsAreNotFlagged:
    items: list = list()
    config: dict = dict()
    buffer: bytearray = bytearray()
    created_at: datetime = datetime.now()
    allowed: frozenset = frozenset([1, 2, 3])
    tags: set = set()


@dataclass
class FieldDefaultFactoryCompliant:
    members: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class FieldDefaultCompliant:
    count: int = field(default=0)
    name: str = field(default="hello")
    derived: int = field(init=False, default=0)


@dataclass
class ClassVarAnnotations:
    count: ClassVar[int] = 5
    items: ClassVar[list] = []
    metadata: ClassVar[dict] = {}


@dataclass
class ClassVarBareForm:
    count: ClassVar = 0


@dataclass
class InitVarAnnotations:
    init_value: InitVar[int] = 1
    init_param: InitVar[str] = "default"


@dataclass
class NoDefaultAnnotatedFields:
    x: float
    y: int
    name: str


@dataclass
class OptionalWithNoneDefault:
    label: Optional[str] = None


@dataclass(frozen=True)
class FrozenDataclassCompliant:
    name: str = ""
    items: list = field(default_factory=list)


@dataclasses.dataclass
class QualifiedDecoratorCompliant:
    name: str = ""
    count: int = 0


def make_default_list():
    return []

@dataclass
class CustomFactoryCompliant:
    items: list = field(default_factory=make_default_list)


@dataclass
class TupleLiteralDefault:
    coords: tuple = (0, 0)


@dataclass
class NestedCallDefault:
    value: str = str(object())


@dataclass
class EmptyDataclass:
    pass


@dataclass
class DataclassWithOnlyMethods:
    name: str

    def get_name(self) -> str:
        return self.name


@dataclass
class ParentDataclass:
    name: str = ""


class NonDataclassChild(ParentDataclass):
    unannotated = 99


# ALL_CAPS names follow the Python constant convention and are intentional class-level attributes
@dataclass
class AllCapsConstants:
    MAX_RETRIES = 5
    DEFAULT_TIMEOUT = 30
    BASE_URL = "https://example.com"


# Names starting with _ are considered intentional private/special attributes
@dataclass
class UnderscorePrefixedAttributes:
    _private = "value"
    __slots__ = ['x', 'y']
    __tablename__ = "users"
    __abstract__ = True
    _PRIVATE_CONST = 42
