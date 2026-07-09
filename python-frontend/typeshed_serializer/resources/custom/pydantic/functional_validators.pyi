from typing import Any, Callable, Literal
from SonarPythonAnalyzerFakeStub import CustomStubBase

FieldValidatorModes = Literal['before', 'after', 'wrap', 'plain']

def field_validator(
    field: str,
    /,
    *fields: str,
    mode: FieldValidatorModes = ...,
    check_fields: bool | None = ...,
    json_schema_input_type: Any = ...,
) -> Callable[..., Any]: ...

def model_validator(
    *,
    mode: Literal['wrap', 'before', 'after'],
) -> Any: ...

class AfterValidator(CustomStubBase):
    def __init__(self, func: Callable[..., Any]) -> None: ...

class BeforeValidator(CustomStubBase):
    def __init__(self, func: Callable[..., Any], json_schema_input_type: Any = ...) -> None: ...

class PlainValidator(CustomStubBase):
    def __init__(self, func: Callable[..., Any], json_schema_input_type: Any = ...) -> None: ...

class WrapValidator(CustomStubBase):
    def __init__(self, func: Callable[..., Any], json_schema_input_type: Any = ...) -> None: ...

class InstanceOf(CustomStubBase): ...

class SkipValidation(CustomStubBase): ...
