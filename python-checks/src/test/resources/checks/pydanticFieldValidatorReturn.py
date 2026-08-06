from pydantic import BaseModel, field_validator
from pydantic.functional_validators import field_validator as fv_direct


class UserModel(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def clean_name(cls, v):  # Noncompliant {{Add a return statement to this Pydantic field validator.}}
    #   ^^^^^^^^^^
        v = v.lower().strip()


class UserModel2(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):  # Noncompliant {{Add a return statement to this Pydantic field validator.}}
        if len(v) < 2:
            raise ValueError("Name too short")
        # Missing return v on the valid path


class EmptyReturn(BaseModel):
    a: str
    b: str

    @field_validator("a", "b")
    @classmethod
    def validate_fields(cls, v):  # Noncompliant {{Add a return statement to this Pydantic field validator.}}
        return


class MultiFieldModel(BaseModel):
    a: str
    b: str

    @field_validator("a", "b")
    @classmethod
    def validate_fields(cls, v):  # Noncompliant {{Add a return statement to this Pydantic field validator.}}
        v = v.strip()


class AliasedImportNoncompliant(BaseModel):
    name: str

    @fv_direct("name")
    @classmethod
    def transform_name(cls, v):  # Noncompliant {{Add a return statement to this Pydantic field validator.}}
        v = v.upper()

# for loop: no return at all — must still be flagged
class NoncompliantForLoopNoReturn(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):  # Noncompliant {{Add a return statement to this Pydantic field validator.}}
        for char in v:
            print(char)

# for loop: no return at all — must still be flagged
class NoncompliantWhileLoopNoReturn(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):  # Noncompliant {{Add a return statement to this Pydantic field validator.}}
        while v:
            print(char)

class CompliantModel(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def clean_name(cls, v):
        return v.lower().strip()


class CompliantConditional(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if len(v) < 2:
            raise ValueError("Name too short")
        return v


class CompliantConditionalBothBranches(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v.startswith("Dr."):
            return v
        else:
            return v.title()


class CompliantAlwaysRaises(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        raise NotImplementedError("Not supported")


class CompliantAliasImport(BaseModel):
    name: str

    @fv_direct("name")
    @classmethod
    def clean_name(cls, v):
        return v.lower()


class CompliantPassBody(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        pass


class CompliantEllipsisBody(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        ...




# Non-BaseModel class should not be flagged even with pydantic decorator
class PlainClass:
    name: str

    @fv_direct("name")
    @classmethod
    def clean_name(cls, v):
        v = v.lower()


class CompliantYield(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        yield v


# with-statement: the WITH_STMT is an end-predecessor in CFG — must not produce a false positive
class CompliantWithStatement(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        with open("/dev/null") as f:
            return f.read() or v


# while loop with return inside: the WHILE_STMT is an end-predecessor in CFG — must not produce a false positive
class CompliantWhileLoop(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        while something:
            return v.strip()

class CompliantIfTrue(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if True:
            return v.strip()


# try/except: CFG falls back to simple check; no return AND raises → not flagged (always raises on both paths)
class CompliantTryExceptAlwaysRaises(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        try:
            int(v)
        except ValueError:
            raise ValueError("Not a number")
        raise ValueError("Should not be a number")


# try/except: CFG falls back; has an explicit return → not flagged
class CompliantTryExceptWithReturn(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        try:
            return v.strip()
        except Exception:
            raise ValueError("Unexpected error")



def not_in_model():
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        v.strip() 
        
# A locally-defined field_validator function (not pydantic) should not be flagged
def my_field_validator(*fields, **kwargs):
    pass

class NotPydanticFieldValidator(BaseModel):
    name: str

    @my_field_validator("name")
    @classmethod
    def clean_name(cls, v):
        v = v.lower()


# for loop: return inside loop — must not produce a false positive
class CompliantForLoopReturn(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        for char in v:
            return char



