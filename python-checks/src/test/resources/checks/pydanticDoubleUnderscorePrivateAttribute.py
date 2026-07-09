from pydantic import BaseModel, PrivateAttr

# =====================
# NON-COMPLIANT CASES
# =====================

class Model(BaseModel):
    __counter: int = PrivateAttr(default=0)  # Noncompliant {{Replace this double underscore prefix with a single underscore to avoid Python name mangling.}}
#   ^^^^^^^^^

class ModelWithMultipleIssues(BaseModel):
    __name: str = PrivateAttr(default="")  # Noncompliant
    __value: int = PrivateAttr(default=0)  # Noncompliant
    _ok: str = PrivateAttr(default="ok")  # Compliant

class ModelWithDunderAndRegular(BaseModel):
    __private: str = PrivateAttr(default="")  # Noncompliant
    name: str  # Compliant: not private at all
    _single: int = PrivateAttr(default=1)  # Compliant

class SubModel(BaseModel):
    __sub_attr: float = PrivateAttr(default=0.0)  # Noncompliant

# =====================
# COMPLIANT CASES
# =====================

class CompliantModel(BaseModel):
    _counter: int = PrivateAttr(default=0)  # Compliant: single underscore prefix
    name: str  # Compliant: regular field, no underscore
    _private: str = PrivateAttr(default="hello")  # Compliant

class ModelWithDunderMethod(BaseModel):
    name: str  # Compliant
    # Dunder methods should not be flagged (they're not private attributes with double underscore prefix)

class RegularClass:
    __attr: int = 0  # Compliant: not a Pydantic model

class ModelWithoutDoubleUnderscore(BaseModel):
    _attr: str = PrivateAttr(default="")  # Compliant: single underscore

# Inheritance from non-Pydantic class should not be flagged
class NonPydanticBase:
    pass

class DerivedFromNonPydantic(NonPydanticBase):
    __attr: int = 0  # Compliant: not a Pydantic model

# =====================
# EDGE CASES
# =====================

class EmptyModel(BaseModel):
    pass  # Compliant: no fields

class ModelWithRegularFields(BaseModel):
    id: int  # Compliant: no underscore
    name: str  # Compliant: no underscore
    _status: str = PrivateAttr(default="active")  # Compliant: single underscore

# Unannotated assignments with PrivateAttr should be flagged
class ModelWithUnannotatedPrivateAttr(BaseModel):
    __counter = PrivateAttr(default=0)  # Noncompliant {{Replace this double underscore prefix with a single underscore to avoid Python name mangling.}}
#   ^^^^^^^^^

class ModelWithMultipleUnannotatedIssues(BaseModel):
    __id = PrivateAttr(default=0)  # Noncompliant
    __label = PrivateAttr(default="")  # Noncompliant
    _ok = PrivateAttr(default=0)  # Compliant: single underscore

# Unannotated plain class variable (not a PrivateAttr call) should not be flagged
class ModelWithUnannotatedPlainVar(BaseModel):
    __plain = 0  # Compliant: not a PrivateAttr call

# Non-pydantic class with unannotated double-underscore PrivateAttr should not be flagged
class RegularClassWithUnannotatedPrivateAttr:
    __counter = PrivateAttr(default=0)  # Compliant: not a Pydantic model
