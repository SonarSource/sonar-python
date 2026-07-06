import pydantic_core
from pydantic_core import to_json, to_jsonable_python


class CustomObject:
    def __init__(self, value):
        self.value = value


def handle_unknown(obj):
    return str(obj)

class CustomObject2(Unknown):
    def __init__(self, value):
        self.value = value


data = {"key": CustomObject(42)}

# =====================
# NON-COMPLIANT CASES
# =====================

result = to_json(data)  # Noncompliant {{Add a "fallback" parameter to this pydantic-core serialization call.}}
result = to_jsonable_python(data)  # Noncompliant

result = to_json(data, indent=2)  # Noncompliant
result = to_jsonable_python(data, exclude=None)  # Noncompliant

result = pydantic_core.to_json(data)  # Noncompliant
result = pydantic_core.to_jsonable_python(data)  # Noncompliant

result = pydantic_core.to_json(CustomObject(42))  # Noncompliant

# =====================
# COMPLIANT CASES
# =====================

result = to_json(data, fallback=handle_unknown)
result = to_jsonable_python(data, fallback=handle_unknown)

result = to_json(data, indent=2, fallback=handle_unknown)
result = to_jsonable_python(data, exclude=None, fallback=str)

result = pydantic_core.to_json(data, fallback=handle_unknown)
result = pydantic_core.to_jsonable_python(data, fallback=handle_unknown)

result = pydantic_core.to_jsonable_python(CustomObject2(42))  

result = pydantic_core.to_json()
def foo(**kwargs):
    result = pydantic_core.to_json(**kwargs) # Coverage

# =====================
# UNRELATED CALLS - should not trigger
# =====================

import other_lib
result = other_lib.to_json(data)

# =====================
# PYDANTIC BASEMODEL EXCEPTION - should not trigger
# =====================

from pydantic import BaseModel

class MyModel(BaseModel):
    value: int

class MySubModel(MyModel):
    name: str

model = MyModel(value=42)
sub_model = MySubModel(value=1, name="test")

result = to_json(model)  # Compliant: pydantic.BaseModel instance
result = to_jsonable_python(model)  # Compliant: pydantic.BaseModel instance
result = to_json(sub_model)  # Compliant: subclass of pydantic.BaseModel
result = to_jsonable_python(sub_model)  # Compliant: subclass of pydantic.BaseModel
result = pydantic_core.to_json(model)  # Compliant: pydantic.BaseModel instance
result = pydantic_core.to_jsonable_python(model)  # Compliant: pydantic.BaseModel instance
