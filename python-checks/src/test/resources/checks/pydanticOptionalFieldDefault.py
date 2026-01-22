from typing import Optional, Union
from pydantic import BaseModel, Field

# =====================
# NON-COMPLIANT CASES
# =====================

class UserModel(BaseModel):
    name: str
    email: Optional[str]  # Noncompliant {{Add an explicit default value to this optional field.}}
#          ^^^^^^^^^^^^^

class ProfileModel(BaseModel):
    bio: str | None  # Noncompliant
#        ^^^^^^^^^^

class SettingsModel(BaseModel):
    theme: Optional[str] = Field(...)  # Noncompliant
#          ^^^^^^^^^^^^^

class ArticleModel(BaseModel):
    title: str
    subtitle: Optional[str]  # Noncompliant
    tags: list[str] | None  # Noncompliant

class DataModel(BaseModel):
    value: Union[str, None]  # Noncompliant

class ComplexModel(BaseModel):
    data: Union[str, int, None]  # Noncompliant

# =====================
# COMPLIANT CASES
# =====================

class UserModelCompliant(BaseModel):
    name: str
    email: Optional[str] = None  # Compliant

class ProfileModelCompliant(BaseModel):
    bio: str | None = None  # Compliant

class SettingsModelCompliant(BaseModel):
    theme: Optional[str] = Field(default=None)  # Compliant
    priority: Optional[int] = Field(default=0)  # Compliant

class RequiredModel(BaseModel):
    required_field: str  # Compliant - not Optional
    another_field: int = Field(...)  # Compliant - not Optional

class ConfigModel(BaseModel):
    timeout: Optional[int] = 30  # Compliant - has default

class FactoryModel(BaseModel):
    items: Optional[list] = Field(default_factory=list)  # Compliant

class RegularClass:
    value: Optional[str]  # Compliant - not a BaseModel

# =====================
# EDGE CASES
# =====================

class ComplexModelCompliant(BaseModel):
    data: Union[str, int, None] = None  # Compliant

class EmptyModel(BaseModel):
    pass  # Compliant - no fields

class OnlyRequiredModel(BaseModel):
    id: int
    name: str 

# =====================
# ADDITIONAL EDGE CASES FOR COVERAGE
# =====================

class NoneLeftModel(BaseModel):
    value: None | str  # Noncompliant

class NestedUnionLeftModel(BaseModel):
    value: None | str | int  # Noncompliant

class EmptyFieldModel(BaseModel):
    value: Optional[str] = Field()  # Compliant - Field() with no ellipsis

class FieldWithValueModel(BaseModel):
    value: Optional[str] = Field(42)  # Compliant - first arg is not ellipsis

class FieldWithKeywordFirstModel(BaseModel):
    value: Optional[str] = Field(default=None)  # Compliant - default is specified

class FieldEllipsisWithDefaultModel(BaseModel):
    value: Optional[str] = Field(..., default=None)  # Compliant

class FieldEllipsisWithFactoryModel(BaseModel):
    value: Optional[list] = Field(..., default_factory=list)  # Compliant

class ModelWithMethods(BaseModel):
    optional_field: Optional[str]  # Noncompliant
    required_field: str

    def some_method(self):
        pass

    @classmethod
    def class_method(cls):
        pass

class MultiNestedModel(BaseModel):
    left_none: None | str  # Noncompliant
    right_none: str | None  # Noncompliant
    deep_left: None | int | str  # Noncompliant
    deep_right1: int | str | None  # Noncompliant
    deep_right2: int | str | str | str  # Compliant

def custom_field():
    return None

class CustomFieldModel(BaseModel):
    value: Optional[str] = custom_field()  # Compliant - not pydantic.Field

class OtherSubscriptionModel(BaseModel):
    value: list[str]  # Compliant
