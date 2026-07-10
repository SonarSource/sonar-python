from pydantic import BaseModel, ConfigDict, Field


# =====================
# NON-COMPLIANT CASES
# =====================

class ModelBothFalse(BaseModel):
    model_config = ConfigDict(
        validate_by_alias=False,  # < 1 {{Also set to "False" here.}}
    #   ^^^^^^^^^^^^^^^^^^^^^^^>
        validate_by_name=False  # Noncompliant {{Enable at least one of "validate_by_alias" or "validate_by_name".}}
    #   ^^^^^^^^^^^^^^^^^^^^^^
    )
    my_field: str = Field(alias='my_alias')


class ModelBothFalseOrderReversed(BaseModel):
    model_config = ConfigDict(
        validate_by_name=False,  # Noncompliant {{Enable at least one of "validate_by_alias" or "validate_by_name".}}
        validate_by_alias=False  # < 1 {{Also set to "False" here.}}
    )
    my_field: str = Field(alias='my_alias')


class ModelBothFalseOtherArgs(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        validate_by_alias=False,  # < 1 {{Also set to "False" here.}}
        validate_by_name=False  # Noncompliant {{Enable at least one of "validate_by_alias" or "validate_by_name".}}
    )
    my_field: str = Field(alias='my_alias')


# Variable holding False
flag_false = False

class ModelBothVarFalse(BaseModel):
    model_config = ConfigDict(
        validate_by_alias=flag_false,  # < 1 {{Also set to "False" here.}}
    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^>
        validate_by_name=flag_false  # Noncompliant {{Enable at least one of "validate_by_alias" or "validate_by_name".}}
    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    my_field: str = Field(alias='my_alias')


class ModelAliasFalseLiteralNameVarFalse(BaseModel):
    model_config = ConfigDict(
        validate_by_alias=False,  # < 1 {{Also set to "False" here.}}
    #   ^^^^^^^^^^^^^^^^^^^^^^^>
        validate_by_name=flag_false  # Noncompliant {{Enable at least one of "validate_by_alias" or "validate_by_name".}}
    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    my_field: str = Field(alias='my_alias')


class ModelAliasVarFalseNameFalseLiteral(BaseModel):
    model_config = ConfigDict(
        validate_by_alias=flag_false,  # < 1 {{Also set to "False" here.}}
    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^>
        validate_by_name=False  # Noncompliant {{Enable at least one of "validate_by_alias" or "validate_by_name".}}
    #   ^^^^^^^^^^^^^^^^^^^^^^
    )
    my_field: str = Field(alias='my_alias')


# =====================
# COMPLIANT CASES
# =====================

class ModelAliasTrue(BaseModel):
    model_config = ConfigDict(
        validate_by_alias=True,
        validate_by_name=False
    )
    my_field: str = Field(alias='my_alias')


class ModelNameTrue(BaseModel):
    model_config = ConfigDict(
        validate_by_alias=False,
        validate_by_name=True
    )
    my_field: str = Field(alias='my_alias')


class ModelBothTrue(BaseModel):
    model_config = ConfigDict(
        validate_by_alias=True,
        validate_by_name=True
    )
    my_field: str = Field(alias='my_alias')


class ModelOnlyAlias(BaseModel):
    model_config = ConfigDict(
        validate_by_alias=False
    )
    my_field: str = Field(alias='my_alias')


class ModelOnlyName(BaseModel):
    model_config = ConfigDict(
        validate_by_name=False
    )
    my_field: str


class ModelEmpty(BaseModel):
    model_config = ConfigDict()
    my_field: str


class ModelNoValidation(BaseModel):
    model_config = ConfigDict(frozen=True)
    my_field: str


class ModelAliasFalseNameMissing(BaseModel):
    model_config = ConfigDict(validate_by_alias=False)
    my_field: str


flag_true = True

class ModelAliasVarTrueNameFalse(BaseModel):
    model_config = ConfigDict(
        validate_by_alias=flag_true,
        validate_by_name=False
    )
    my_field: str = Field(alias='my_alias')


class ModelBothVarTrue(BaseModel):
    model_config = ConfigDict(
        validate_by_alias=flag_true,
        validate_by_name=flag_true
    )
    my_field: str = Field(alias='my_alias')


# ConfigDict is always from pydantic, so this is still flagged even in non-BaseModel classes
class NotPydanticSubclass:
    model_config = ConfigDict(
        validate_by_alias=False,  # < 1 {{Also set to "False" here.}}
        validate_by_name=False  # Noncompliant {{Enable at least one of "validate_by_alias" or "validate_by_name".}}
    )


from some_other_lib import ConfigDict as OtherConfigDict

class NotFlaggedNonPydanticConfigDict:
    model_config = OtherConfigDict(
        validate_by_alias=False,
        validate_by_name=False
    )
