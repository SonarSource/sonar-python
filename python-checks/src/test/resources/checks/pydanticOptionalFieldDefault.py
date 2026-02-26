from typing import Optional, Union
from pydantic import BaseModel, Field

# =====================
# NON-COMPLIANT CASES
# =====================

class UserModel(BaseModel):
    name: str
    email: Optional[str]  # Noncompliant {{Add an explicit default value to this optional field.}}
#          ^^^^^^^^^^^^^

class SettingsModel(BaseModel):
    theme: Optional[str] = Field(...)  # Noncompliant
#          ^^^^^^^^^^^^^

class ModelWithMethods(BaseModel):
    optional_field: Optional[str]  # Noncompliant
    required_field: str

    def some_method(self):
        pass

# =====================
# COMPLIANT CASES
# =====================

class UserModelCompliant(BaseModel):
    name: str
    email: Optional[str] = None

class ProfileModelCompliant(BaseModel):
    bio: str | None = None

class SettingsModelCompliant(BaseModel):
    theme: Optional[str] = Field(default=None)
    priority: Optional[int] = Field(default=0)

class RequiredModel(BaseModel):
    required_field: str
    another_field: int = Field(...)

class ConfigModel(BaseModel):
    timeout: Optional[int] = 30

class FactoryModel(BaseModel):
    items: Optional[list] = Field(default_factory=list)

class RegularClass:
    value: Optional[str]

class BitwiseOrNoneCompliant(BaseModel):
    reason: int | None

class BitwiseOrNoneWithFieldEllipsisCompliant(BaseModel):
    bio: str | None = Field(...)

class BitwiseOrNoneWithFieldDefaultCompliant(BaseModel):
    title: str | None = Field(default=None)

class NoneLeftCompliant(BaseModel):
    description: None | str

class NoneLeftWithFieldEllipsisCompliant(BaseModel):
    value: None | str = Field(...)

class UnionWithNoneCompliant(BaseModel):
    data: Union[str, None]

class UnionWithFieldEllipsisCompliant(BaseModel):
    value: Union[str, None] = Field(...)

# =====================
# EDGE CASES
# =====================

class ComplexModelCompliant(BaseModel):
    data: Union[str, int, None] = None

class EmptyModel(BaseModel):
    pass

class OnlyRequiredModel(BaseModel):
    id: int
    name: str

class EmptyFieldModel(BaseModel):
    value: Optional[str] = Field()

class FieldWithValueModel(BaseModel):
    value: Optional[str] = Field(42)

class FieldWithKeywordFirstModel(BaseModel):
    value: Optional[str] = Field(default=None)

class FieldEllipsisWithDefaultModel(BaseModel):
    value: Optional[str] = Field(..., default=None)

class FieldEllipsisWithFactoryModel(BaseModel):
    value: Optional[list] = Field(..., default_factory=list)

class MultiNoneFieldCompliant(BaseModel):
    deep_left: None | int | str
    deep_right: int | str | None
    no_none: int | str | str

def custom_field():
    return None

class CustomFieldModel(BaseModel):
    value: Optional[str] = custom_field()

class OtherSubscriptionModel(BaseModel):
    value: list[str]
