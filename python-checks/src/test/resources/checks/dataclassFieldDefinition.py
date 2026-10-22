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


def make_method(operation):
    def method(self, *args):
        return operation(*args)
    return method


def method(self):
    return 42


@dataclass
class GeneratedMethods:
    direct = method
    generated = make_method(sum)
    anonymous = lambda self: 42
    count = 42  # Noncompliant
    items = list()  # Noncompliant


def nested_generated_methods():
    @dataclass
    class Operations:
        bitwise_and = make_method(sum)
        bitwise_or = make_method(sum)
        count = 42  # Noncompliant


def make_data():
    def method(self):
        return 42
    return 42


def make_pair():
    return method, 42


async def make_coroutine():
    return method


def make_generator():
    yield 42
    return method


class CallableData:
    def __call__(self):
        return 42


@dataclass
class FactoryDataStillRequiresAnnotations:
    data = make_data()  # Noncompliant
    pair = make_pair()  # Noncompliant
    coroutine = make_coroutine()  # Noncompliant
    generator = make_generator()  # Noncompliant
    callable_data = CallableData()  # Noncompliant
    field_default = field(default=42)  # Noncompliant


def make_lambda():
    return lambda self: 42


@dataclass
class LambdaFactory:
    generated = make_lambda()
    parenthesized = (lambda self: 42)


def make_mixed_value(as_method=False):
    if as_method:
        return lambda self: 42
    return 42


def make_rebound_value():
    return lambda self: 42


make_rebound_value = int


def make_generator_in_call():
    print((yield 42))
    return lambda self: 42


@dataclass
class ReviewRegressionControls:
    mixed = make_mixed_value()  # Noncompliant
    rebound = make_rebound_value()  # Noncompliant
    generator = make_generator_in_call()  # Noncompliant


def make_optional_method(as_method=False):
    if as_method:
        return method


def make_method_or_none(as_method=False):
    if as_method:
        return method
    return


def make_no_value():
    pass


def make_redefined_value():
    return method


def make_redefined_value():
    return 0


def make_generator_with_yield_from():
    print((yield from [42]))
    return method


def make_generator_in_default():
    def nested(value=(yield 42)):
        return value
    return nested


def make_generator_in_lambda_default():
    return lambda self, value=(yield 42): value


def replace_factory(factory):
    return int


@replace_factory
def make_decorated_value():
    return method


@dataclass
class AmbiguousFactoryResults:
    implicit_none = make_optional_method()  # Noncompliant
    bare_return = make_method_or_none()  # Noncompliant
    no_return = make_no_value()  # Noncompliant
    redefined = make_redefined_value()  # Noncompliant
    delegated_generator = make_generator_with_yield_from()  # Noncompliant
    generator_default = make_generator_in_default()  # Noncompliant
    generator_lambda_default = make_generator_in_lambda_default()  # Noncompliant
    decorated = make_decorated_value()  # Noncompliant


def make_alternative_methods(first=True):
    if first:
        return method
    else:
        return lambda self: 42


def make_generator_method():
    def generated(self):
        yield 42
    return generated


def make_lambda_generator_method():
    return lambda self: (yield 42)


def make_method_with_nested_class():
    class Nested:
        def generated(self):
            yield 42
    return method


@dataclass
class MoreGeneratedMethods:
    alternatives = make_alternative_methods()
    generator_method = make_generator_method()
    lambda_generator_method = make_lambda_generator_method()
    nested_class = make_method_with_nested_class()
