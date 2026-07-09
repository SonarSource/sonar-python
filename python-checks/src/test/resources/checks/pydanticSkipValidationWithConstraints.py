from typing import Annotated
from pydantic import BaseModel, SkipValidation, Field
from pydantic import StringConstraints, AfterValidator, BeforeValidator, PlainValidator, WrapValidator

# =====================
# NON-COMPLIANT CASES
# =====================

class Model(BaseModel):
    # SkipValidation combined with Field in the same Annotated
    value: Annotated[int, SkipValidation, Field(gt=0)]  # Noncompliant {{Remove either "SkipValidation" or the validation constraints from this annotation.}}
    #                     ^^^^^^^^^^^^^^ 1^^^^^^^^^^^< {{Validation constraint is set here.}}

    constraint = Field(gt=0)
    #            ^^^^^^^^^^^> {{Validation constraint is set here.}}
    value: Annotated[int, SkipValidation, constraint]  # Noncompliant {{Remove either "SkipValidation" or the validation constraints from this annotation.}}
    #                     ^^^^^^^^^^^^^^ 
    # Field constraint in inner Annotated, SkipValidation in outer
    other: Annotated[Annotated[int, Field(gt=0)], SkipValidation]  # Noncompliant
    #                               ^^^^^^^^^^^> {{Validation constraint is set here.}}
    #                                             ^^^^^^^^^^^^^^@-1 1

    # SkipValidation with StringConstraints
    name: Annotated[str, SkipValidation, StringConstraints(min_length=1)]  # Noncompliant
    #                    ^^^^^^^^^^^^^^ 1^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Validation constraint is set here.}}

    # SkipValidation combined with AfterValidator
    validated: Annotated[int, SkipValidation, AfterValidator(lambda x: x)]  # Noncompliant
    #                         ^^^^^^^^^^^^^^ 1^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Validation constraint is set here.}}

    # SkipValidation combined with BeforeValidator (constraint before SkipValidation)
    pre_validated: Annotated[int, BeforeValidator(lambda x: x), SkipValidation]  # Noncompliant
    #                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{Validation constraint is set here.}}
    #                                                           ^^^^^^^^^^^^^^@-1 1

    # SkipValidation in outer, StringConstraints in nested inner Annotated
    nested_str: Annotated[Annotated[str, StringConstraints(min_length=1)], SkipValidation]  # Noncompliant
    #                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{Validation constraint is set here.}}
    #                                                                      ^^^^^^^^^^^^^^@-1 1

    # SkipValidation combined with PlainValidator
    plain: Annotated[int, SkipValidation, PlainValidator(lambda x: x)]  # Noncompliant
    #                     ^^^^^^^^^^^^^^ 1^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Validation constraint is set here.}}

    # SkipValidation combined with WrapValidator
    wrapped: Annotated[int, SkipValidation, WrapValidator(lambda x, handler: handler(x))]  # Noncompliant
    #                       ^^^^^^^^^^^^^^ 1^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Validation constraint is set here.}}

    # Field with min_length kwarg + SkipValidation
    min_len_field: Annotated[str, SkipValidation, Field(min_length=1)]  # Noncompliant
    #                             ^^^^^^^^^^^^^^ 1^^^^^^^^^^^^^^^^^^^< {{Validation constraint is set here.}}

    # Field with multiple constraint kwargs + SkipValidation
    multi_constraint: Annotated[int, SkipValidation, Field(ge=0, le=100)]  # Noncompliant
    #                                ^^^^^^^^^^^^^^ 1^^^^^^^^^^^^^^^^^^^< {{Validation constraint is set here.}}

    # Field with constraint kwarg mixed with metadata kwargs + SkipValidation: still a constraint
    mixed_field: Annotated[int, SkipValidation, Field(default=0, gt=0)]  # Noncompliant
    #                           ^^^^^^^^^^^^^^ 1^^^^^^^^^^^^^^^^^^^^^^< {{Validation constraint is set here.}}


# =====================
# COMPLIANT CASES
# =====================

class CompliantModel(BaseModel):
    # SkipValidation alone with a subscript type - OK
    trusted: SkipValidation[int]

    # SkipValidation as only annotation in Annotated - OK
    also_trusted: Annotated[int, SkipValidation]

    # Field alone without SkipValidation - OK
    constrained: Annotated[int, Field(gt=0)]

    # StringConstraints alone - OK
    string_field: Annotated[str, StringConstraints(min_length=1)]

    # Nested Annotated without SkipValidation - OK
    nested: Annotated[Annotated[int, Field(gt=0)], Field(lt=100)]

    # AfterValidator alone - OK
    after: Annotated[int, AfterValidator(lambda x: x)]

    # Plain type annotation - OK
    plain_type: int

    # Field with a default - OK
    optional: Annotated[int, Field(default=0)]

    # Field with only default + SkipValidation: no validation constraint, so OK
    default_trusted: Annotated[int, SkipValidation, Field(default=0)]

    # Field with only alias + SkipValidation: alias is metadata, not a constraint
    alias_trusted: Annotated[str, SkipValidation, Field(alias="x")]

    # Field with only description + SkipValidation: description is metadata, not a constraint
    desc_trusted: Annotated[str, SkipValidation, Field(description="some description")]

    # Field with only serialization_alias + SkipValidation
    ser_alias_trusted: Annotated[str, SkipValidation, Field(serialization_alias="x")]

    # Field with only validation_alias + SkipValidation
    val_alias_trusted: Annotated[str, SkipValidation, Field(validation_alias="x")]

    # Field with default_factory + SkipValidation: no constraint, OK
    factory_trusted: Annotated[list, SkipValidation, Field(default_factory=list)]

    # Field with title + SkipValidation: metadata only
    title_trusted: Annotated[str, SkipValidation, Field(title="My Field")]

    # Field with exclude + SkipValidation: serialization flag, not a constraint
    exclude_trusted: Annotated[str, SkipValidation, Field(exclude=True)]

    # Behavioral flags: with SkipValidation, no validation runs so these have no effect regardless of value
    strict_true: Annotated[int, SkipValidation, Field(strict=True)]
    strict_false: Annotated[int, SkipValidation, Field(strict=False)]
    allow_inf_nan_true: Annotated[float, SkipValidation, Field(allow_inf_nan=True)]
    allow_inf_nan_false: Annotated[float, SkipValidation, Field(allow_inf_nan=False)]
    coerce_str: Annotated[str, SkipValidation, Field(coerce_numbers_to_str=True)]
    fail_fast_field: Annotated[int, SkipValidation, Field(fail_fast=True)]
    discriminator_field: Annotated[int, SkipValidation, Field(discriminator="type")]


# Regular class (not a Pydantic model) - should not trigger
class RegularClass:
    value: Annotated[int, SkipValidation, Field(gt=0)]


# Standalone annotations outside any class - should not trigger
standalone: Annotated[int, SkipValidation, Field(gt=0)]


def process(value: Annotated[int, SkipValidation, Field(gt=0)]) -> None:
    pass
