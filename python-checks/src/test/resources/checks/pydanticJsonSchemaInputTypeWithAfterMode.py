from pydantic import BaseModel, field_validator

# =====================
# NON-COMPLIANT CASES
# =====================

class ModelAfterMode(BaseModel):
    a: str

    @field_validator('a', mode='after', json_schema_input_type=str)  # Noncompliant {{Remove "json_schema_input_type" or change the validator mode to "before", "plain", or "wrap".}}
    #                                   ^^^^^^^^^^^^^^^^^^^^^^
    #                          ^^^^^^^@-1< {{mode is set here.}}
    @classmethod
    def validate_a(cls, v):
        return v


class ModelDefaultMode(BaseModel):
    a: str

    @field_validator('a', json_schema_input_type=str)  # Noncompliant {{Remove "json_schema_input_type" or change the validator mode to "before", "plain", or "wrap".}}
    @classmethod
    def validate_a_default(cls, v):
        return v


class ModelAfterModeMultipleFields(BaseModel):
    a: str
    b: int

    @field_validator('a', 'b', mode='after', json_schema_input_type=str)  # Noncompliant {{Remove "json_schema_input_type" or change the validator mode to "before", "plain", or "wrap".}}
    @classmethod
    def validate_ab(cls, v):
        return v


class ModelModeAfterExplicit(BaseModel):
    a: str

    @field_validator('a', json_schema_input_type=int, mode='after')  # Noncompliant {{Remove "json_schema_input_type" or change the validator mode to "before", "plain", or "wrap".}}
    @classmethod
    def validate_a2(cls, v):
        return v

# =====================
# COMPLIANT CASES
# =====================

class ModelBeforeMode(BaseModel):
    a: str

    @field_validator('a', mode='before', json_schema_input_type=str)
    @classmethod
    def validate_a(cls, v):
        return v


class ModelPlainMode(BaseModel):
    a: str

    @field_validator('a', mode='plain', json_schema_input_type=str)
    @classmethod
    def validate_a(cls, v):
        return v


class ModelWrapMode(BaseModel):
    a: str

    @field_validator('a', mode='wrap', json_schema_input_type=str)
    @classmethod
    def validate_a(cls, v, handler):
        return handler(v)


class ModelAfterModeNoJsonSchemaInputType(BaseModel):
    a: str

    @field_validator('a', mode='after')
    @classmethod
    def validate_a(cls, v):
        return v


class ModelNoJsonSchemaInputType(BaseModel):
    a: str

    @field_validator('a')
    @classmethod
    def validate_a(cls, v):
        return v


class ModelModeIsVariable(BaseModel):
    a: str

    VALIDATOR_MODE = 'after'

    @field_validator('a', mode=VALIDATOR_MODE, json_schema_input_type=str)
    @classmethod
    def validate_a(cls, v):
        return v


# A function with the same name but no pydantic import should not be flagged
def field_validator(*fields, **kwargs):
    pass

@field_validator('a', mode='after', json_schema_input_type=str)
def validate_local(cls, v):
    return v


# =====================
# DIRECT MODULE IMPORT
# =====================

from pydantic.functional_validators import field_validator as fv_direct  # noqa: E402

class ModelDirectImportNoncompliant(BaseModel):
    a: str

    @fv_direct('a', mode='after', json_schema_input_type=str)  # Noncompliant {{Remove "json_schema_input_type" or change the validator mode to "before", "plain", or "wrap".}}
    @classmethod
    def validate_a(cls, v):
        return v


class ModelDirectImportCompliant(BaseModel):
    a: str

    @fv_direct('a', mode='before', json_schema_input_type=str)
    @classmethod
    def validate_a(cls, v):
        return v


# =====================
# ALIASED IMPORT
# =====================

from pydantic import field_validator as fv_alias  # noqa: E402

class ModelAliasedImportNoncompliant(BaseModel):
    a: str

    @fv_alias('a', json_schema_input_type=str)  # Noncompliant {{Remove "json_schema_input_type" or change the validator mode to "before", "plain", or "wrap".}}
    @classmethod
    def validate_a(cls, v):
        return v


class ModelAliasedImportCompliant(BaseModel):
    a: str

    @fv_alias('a', mode='before', json_schema_input_type=str)
    @classmethod
    def validate_a(cls, v):
        return v
