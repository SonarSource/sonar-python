from pydantic import BaseModel, ConfigDict


# =====================
# NON-COMPLIANT CASES
# =====================

class Base1(BaseModel):
    model_config = ConfigDict(str_to_lower=True)

class Base2(BaseModel):
    model_config = ConfigDict(str_to_upper=True)

class Model(Base1, Base2):  # Noncompliant {{Refactor this Pydantic model to avoid multiple inheritance with conflicting configurations.}}
#     ^^^^^
    x: str


class Base3(BaseModel):
    model_config = ConfigDict(frozen=True)

class Base4(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

class ModelThreeBases(Base1, Base3, Base4):  # Noncompliant
#     ^^^^^^^^^^^^^^^
    y: int


# Multiple noncompliant models in the same file
class AnotherModel(Base1, Base2):  # Noncompliant
#     ^^^^^^^^^^^^
    z: str


# =====================
# COMPLIANT CASES
# =====================

# Only one base has model_config
class Base5(BaseModel):
    pass

class ModelOneConfigBase(Base1, Base5):  # Compliant - only Base1 has model_config
    a: str


# Neither base defines model_config
class BaseNoConfig1(BaseModel):
    x: int

class BaseNoConfig2(BaseModel):
    y: str

class ModelNoConfig(BaseNoConfig1, BaseNoConfig2):  # Compliant
    z: float


# Single inheritance with config
class ModelSingleInheritance(Base1):  # Compliant
    b: int


# Single base with config, no multiple inheritance conflict
class ModelWithOwnConfig(Base1):  # Compliant
    model_config = ConfigDict(str_to_lower=True)
    c: str


# Not a Pydantic BaseModel subclass
class NotPydantic:
    model_config = ConfigDict(str_to_lower=True)

class AlsoNotPydantic:
    model_config = ConfigDict(str_to_upper=True)

class NotPydanticMultiple(NotPydantic, AlsoNotPydantic):  # Compliant - not a Pydantic model
    pass


# BaseModel itself is not flagged
class DirectBaseModel(BaseModel):
    pass


# Only the current class itself defines model_config (not multiple bases)
class ModelConfigOnChild(Base1):  # Compliant - only one base (Base1) has model_config
    model_config = ConfigDict(frozen=True)
    d: int


# =====================
# INDIRECT INHERITANCE (MRO-based detection)
# =====================

# Base that inherits model_config from Base1 without redefining it
class IntermediateBase(Base1):
    pass

# Both IntermediateBase (via Base1) and Base2 define model_config — conflict
class ModelIndirect(IntermediateBase, Base2):  # Noncompliant {{Refactor this Pydantic model to avoid multiple inheritance with conflicting configurations.}}
#     ^^^^^^^^^^^^^
    e: str

# Two levels of indirection
class DeepBase(IntermediateBase):
    pass

class ModelDeep(DeepBase, Base2):  # Noncompliant
#     ^^^^^^^^^
    f: str

# Both direct bases ultimately source model_config from the same class (Base1) — no conflict
class ModelSameSource(IntermediateBase, Base1):  # Compliant - same model_config source
    g: str

# Intermediate base that does NOT carry a config, paired with a base that does — only one definer
class IntermediateNoConfig(Base5):
    pass

class ModelOneIndirectConfig(IntermediateNoConfig, Base1):  # Compliant - only Base1 defines model_config
    h: str


# =====================
# EDGE CASES
# =====================

# Keyword argument in the base list must be ignored (not treated as a positional base)
class ModelWithKeywordArg(Base1, Base2, metaclass=type):  # Noncompliant
#     ^^^^^^^^^^^^^^^^^^^
    i: str

# Diamond inheritance: A and B both inherit from Base1 (which defines model_config).
# The BFS visited-guard must deduplicate Base1 so it is only counted once — no conflict.
class DiamondA(Base1):
    pass

class DiamondB(Base1):
    pass

class ModelDiamond(DiamondA, DiamondB):  # Compliant - single model_config source (Base1)
    j: str

# A base expression that does not resolve to a ClassType (e.g. a call) must be skipped
def make_base():
    return Base1

class ModelWithCallBase(make_base(), Base2):  # Compliant - make_base() is not a ClassType
    k: str


# =====================
# EXCEPTION: child defines own model_config
# =====================

# Child defines its own model_config with multiple conflicting bases — compliant (exception)
class ModelOwnConfigMultiBases(Base1, Base2):  # Compliant - child defines its own model_config
    model_config = ConfigDict(frozen=True)
    x: str

# Same exception with three bases
class ModelOwnConfigThreeBases(Base1, Base2, Base3):  # Compliant - child defines its own model_config
    model_config = ConfigDict(frozen=True)
    y: int

# Exception holds even with indirect/MRO-based conflict
class ModelOwnConfigIndirect(IntermediateBase, Base2):  # Compliant - child defines its own model_config
    model_config = ConfigDict(str_to_lower=True)
    z: str


# =====================
# UNRESOLVED SUPERCLASSES
# =====================

# One superclass cannot be resolved to a ClassType (unknown import) — the rule skips it.
# Only Base1 contributes a model_config definer, so no conflict is raised.
class ModelOneUnresolved(UnknownBase, Base1):  # Compliant - UnknownBase is unresolved
    m: str

# Both superclasses are unresolved — no ClassType definer is found at all, no issue raised.
class ModelBothUnresolved(UnknownBase1, UnknownBase2):  # Compliant - both are unresolved
    n: str


# =====================
# KNOWN FALSE POSITIVE
# =====================

# Two bases define identical model_config values — no actual conflict exists, but the rule
# flags it anyway because it does not compare ConfigDict values, only structural presence.
# Pydantic's merge behavior is still non-MRO regardless of whether values currently agree,
# and they can diverge independently in future code changes.
class BaseIdentical1(BaseModel):
    model_config = ConfigDict(frozen=True)

class BaseIdentical2(BaseModel):
    model_config = ConfigDict(frozen=True)

# Known false positive: configs are identical but the rule flags it anyway because it does
# not compare ConfigDict values — only their structural presence.
class ModelIdenticalConfig(BaseIdentical1, BaseIdentical2):  # Noncompliant
#     ^^^^^^^^^^^^^^^^^^^^
    l: str
