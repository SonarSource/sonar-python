import dataclasses
import pydantic.dataclasses as pydantic_dataclasses
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Annotated, Union


# =====================
# NON-COMPLIANT CASES
# =====================

@dataclasses.dataclass
class User:
    name: str


class Foo(BaseModel):  # Noncompliant {{Explicitly set 'revalidate_instances' in this Pydantic model's configuration.}}
#     ^^^
    user: User
#   ^^^^< {{The dataclass-typed field is defined here.}}


@dataclasses.dataclass
class Address:
    street: str


class Bar(BaseModel):  # Noncompliant
#     ^^^
    address: Address
#   ^^^^^^^< {{The dataclass-typed field is defined here.}}


# model_config present but without revalidate_instances
class Baz(BaseModel):  # Noncompliant
#     ^^^
    model_config = ConfigDict(frozen=True)
    user: User
#   ^^^^< {{The dataclass-typed field is defined here.}}


# multiple dataclass fields - one issue per class, secondary on each field
class MultiField(BaseModel):  # Noncompliant
#     ^^^^^^^^^^
    user: User
#   ^^^^< {{The dataclass-typed field is defined here.}}
    address: Address
#   ^^^^^^^< {{The dataclass-typed field is defined here.}}


# =====================
# FINDING 2: WRAPPED ANNOTATION TYPES (NON-COMPLIANT)
# =====================

class WrappedOptional(BaseModel):  # Noncompliant
#     ^^^^^^^^^^^^^^^
    user: Optional[User]
#   ^^^^< {{The dataclass-typed field is defined here.}}

class WrappedList(BaseModel):  # Noncompliant
#     ^^^^^^^^^^^
    users: List[User]
#   ^^^^^< {{The dataclass-typed field is defined here.}}

class WrappedAnnotated(BaseModel):  # Noncompliant
#     ^^^^^^^^^^^^^^^^
    user: Annotated[User, 'meta']
#   ^^^^< {{The dataclass-typed field is defined here.}}

class WrappedUnion(BaseModel):  # Noncompliant
#     ^^^^^^^^^^^^
    user: Union[User, None]
#   ^^^^< {{The dataclass-typed field is defined here.}}

class WrappedNestedGeneric(BaseModel):  # Noncompliant
#     ^^^^^^^^^^^^^^^^^^^^
    data: dict[str, User]
#   ^^^^< {{The dataclass-typed field is defined here.}}


# =====================
# PYDANTIC DATACLASS AS FIELD TYPE
# =====================

@pydantic_dataclasses.dataclass
class PydanticUser:
    name: str


class WithPydanticDataclass(BaseModel):  # already validated
    user: PydanticUser


# =====================
# COMPLIANT CASES
# =====================

# revalidate_instances='always' via ConfigDict
class Compliant(BaseModel):
    model_config = ConfigDict(revalidate_instances='always')
    user: User


# ConfigDict assigned to a non-standard variable name — still detected by callee type
class CompliantNonStandardVarName(BaseModel):
    my_config = ConfigDict(revalidate_instances='always')
    user: User


# revalidate_instances='always' with other options
class CompliantWithOtherOptions(BaseModel):
    model_config = ConfigDict(frozen=True, revalidate_instances='always')
    user: User


# revalidate_instances='never' — any explicit value is acceptable
class CompliantNever(BaseModel):
    model_config = ConfigDict(revalidate_instances='never')
    user: User


# revalidate_instances='subclass-instances' — any explicit value is acceptable
class CompliantSubclassInstances(BaseModel):
    model_config = ConfigDict(revalidate_instances='subclass-instances')
    user: User


# class keyword argument syntax — 'always'
class CompliantKwargAlways(BaseModel, revalidate_instances='always'):
    user: User


# class keyword argument syntax — 'never' (explicit opt-out)
class CompliantKwargNever(BaseModel, revalidate_instances='never'):
    user: User


# class keyword argument syntax — 'subclass-instances'
class CompliantKwargSubclass(BaseModel, revalidate_instances='subclass-instances'):
    user: User


# No dataclass-typed fields — no issue
class NoDataclassField(BaseModel):
    name: str
    age: int


# Field typed as a plain class (not a dataclass) — no issue
class PlainClass:
    name: str

class WithPlainClass(BaseModel):
    obj: PlainClass


# Not a BaseModel subclass — no issue
@dataclasses.dataclass
class StandaloneDataclass:
    value: int


# Not a BaseModel — dataclass with dataclass field
@dataclasses.dataclass
class Container:
    user: User


# =====================
# FINDING 1: DICT-LITERAL model_config (COMPLIANT)
# =====================

# dict-literal with revalidate_instances — any explicit value is acceptable
class CompliantDictLiteralAlways(BaseModel):
    model_config = {'revalidate_instances': 'always'}
    user: User

class CompliantDictLiteralNever(BaseModel):
    model_config = {'revalidate_instances': 'never'}
    user: User

class CompliantDictLiteralWithOtherKeys(BaseModel):
    model_config = {'frozen': True, 'revalidate_instances': 'always'}
    user: User


# =====================
# FINDING 2: WRAPPED ANNOTATION TYPES (COMPLIANT)
# =====================

class CompliantWrappedOptional(BaseModel):
    model_config = ConfigDict(revalidate_instances='always')
    user: Optional[User]

class CompliantWrappedList(BaseModel):
    model_config = ConfigDict(revalidate_instances='always')
    users: List[User]


# =====================
# FINDING 3: INHERITED revalidate_instances (COMPLIANT)
# =====================

class BaseWithConfigDictReval(BaseModel):
    model_config = ConfigDict(revalidate_instances='always')

class ChildOfConfigDictBase(BaseWithConfigDictReval):  # Compliant — inherits via ConfigDict
    user: User

class BaseWithKwargReval(BaseModel, revalidate_instances='always'):
    pass

class ChildOfKwargBase(BaseWithKwargReval):  # Compliant — inherits via keyword arg
    user: User

# Multi-hop: grandchild inherits from a child that inherits revalidate_instances
class GrandchildOfConfigDictBase(ChildOfConfigDictBase):  # Compliant — inherits transitively
    address: Address


# =====================
# FINDING 4: MODULE-LEVEL / ALIASED ConfigDict VARIABLE (COMPLIANT)
# =====================

# module-level variable holding ConfigDict with revalidate_instances
COMMON_CONFIG = ConfigDict(revalidate_instances='always')

class CompliantModuleLevelConfigVar(BaseModel):
    model_config = COMMON_CONFIG
    user: User


# chained alias: A = ConfigDict(...); B = A; model_config = B
ALIAS_A = ConfigDict(revalidate_instances='always')
ALIAS_B = ALIAS_A

class CompliantChainedAlias(BaseModel):
    model_config = ALIAS_B
    user: User


# module-level dict literal with revalidate_instances
DICT_CONFIG = {'revalidate_instances': 'always'}

class CompliantModuleLevelDictVar(BaseModel):
    model_config = DICT_CONFIG
    user: User


# module-level variable WITHOUT revalidate_instances — still noncompliant
BAD_CONFIG = ConfigDict(frozen=True)

class NoncompliantModuleLevelConfigVar(BaseModel):  # Noncompliant
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    model_config = BAD_CONFIG
    user: User
#   ^^^^< {{The dataclass-typed field is defined here.}}


# =====================
# EDGE CASES
# =====================

# Class with no fields — no issue
class EmptyModel(BaseModel):
    pass


# BaseModel with only class variables — no issue (no AnnotatedAssignment without annotation being a dataclass)
class ModelWithClassVar(BaseModel):
    name: str


# Cross-file dataclass fields cannot be detected — treated as unknown type, no issue raised
# (no test case needed here since we only detect same-file dataclasses)

# Unresolved type annotation — no issue (unknown type, conservative)
class ModelWithUnresolved(BaseModel):
    user: UnknownType


# =====================
# COVERAGE: ADDITIONAL EDGE CASES
# =====================

# PEP-604 union syntax (X | Y) — noncompliant
class PEP604Union(BaseModel):  # Noncompliant
#     ^^^^^^^^^^^
    user: User | None
#   ^^^^< {{The dataclass-typed field is defined here.}}


# PEP-604 union syntax — compliant with revalidate_instances set
class CompliantPEP604Union(BaseModel):
    model_config = ConfigDict(revalidate_instances='always')
    user: User | None


# PEP-604 union where right operand is the dataclass — noncompliant
class PEP604UnionRightOperand(BaseModel):  # Noncompliant
#     ^^^^^^^^^^^^^^^^^^^^^^^
    user: None | User
#   ^^^^< {{The dataclass-typed field is defined here.}}


# model_config assigned something that is not a ConfigDict call, dict literal, or variable reference —
# no revalidate_instances found, so issue IS raised
def make_config():
    return ConfigDict(revalidate_instances='always')

class ModelWithCallRHS(BaseModel):  # Noncompliant
#     ^^^^^^^^^^^^^^^^
    model_config = make_config()
    user: User
#   ^^^^< {{The dataclass-typed field is defined here.}}


# Ambiguous Name RHS (assigned in multiple places) — conservative, treated as no config
AMBIGUOUS_CONFIG = ConfigDict(revalidate_instances='always')
AMBIGUOUS_CONFIG = ConfigDict(frozen=True)

class ModelWithAmbiguousConfig(BaseModel):  # Noncompliant
#     ^^^^^^^^^^^^^^^^^^^^^^^^
    model_config = AMBIGUOUS_CONFIG
    user: User
#   ^^^^< {{The dataclass-typed field is defined here.}}


# Annotation that is not a Name, SubscriptionExpression, or BinaryExpression — no issue
class ModelWithLiteralAnnotation(BaseModel):
    value: 42
