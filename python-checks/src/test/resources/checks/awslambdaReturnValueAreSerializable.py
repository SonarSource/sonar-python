import dataclasses
import datetime
import re
from decimal import Decimal
from io import StringIO
import json

# Custom class for testing
@dataclasses.dataclass
class User:
    name: str
    age: int

    def to_dict(self) -> dict:
        return {"name": self.name, "age": self.age}


class CustomObject:
    def __init__(self, value):
        self.value = value


class Serializable:
    def __init__(self, value):
        self.value = value

    def __json__(self):
        ...

# Non-compliant Lambda handlers - returning non-JSON serializable values
def datetime_lambda_handler(event, context):
    return {
        "message": "Request processed",
        "timestamp": datetime.datetime.now(),  # Noncompliant {{Fix the return value to be JSON serializable.}}
        #            ^^^^^^^^^^^^^^^^^^^^^^^
    }


def set_lambda_handler(event, context):
    data_set = {1, 2, 3, 4, 5}
    #          ^^^^^^^^^^^^^^^> {{The non-serializable value is set here.}}
    return {
        "data": data_set,  # Noncompliant
        #       ^^^^^^^^
        "test": set([2]),  # Noncompliant
    }


def custom_object_lambda_handler(event, context):
    alice_non_comp = User("Alice", 30)
    return {
        "user": alice_non_comp  # Noncompliant
    }


def custom_class_lambda_handler(event, context):
    obj = CustomObject("test")
    return obj  # Noncompliant


def list_with_datetime_lambda_handler(event, context):
    return [
        {"id": 1, "created": datetime.datetime.now()},  # Noncompliant
        {"id": 2, "created": datetime.datetime.now()},  # Noncompliant
    ]


def foo(): ...


def mixed_types_lambda_handler(event, context):
    return {
        "tuple": (3j, "str"),  # Noncompliant
        "complex_literal": 3j,  # Noncompliant
        "complex": complex(3, 5),  # Noncompliant
        "fun": foo,  # Noncompliant
        "file_like": StringIO("test"),  # Noncompliant
        "decimal": Decimal(1),  # Noncompliant
        "regex": re.compile(r"\w"),  # Noncompliant
        "user": User("Bob", 25),  # Noncompliant
        "bytes": bytes("test", "utf-8"),  # Noncompliant
        "byte_array": bytearray([1, 2]),  # Noncompliant
        "frozen_set": frozenset({1, 3}),  # Noncompliant
        datetime.datetime.now(): [  # Noncompliant
            User("Alice", 30),  # Noncompliant
        ],
        "file":open("test"), # Noncompliant
        "metadata": {
            "nested": {"urgent", "important"},  # Noncompliant
        },
    }


def conditional_return_lambda_handler(event, context):
    if event.get("type") == "error":
        return {"error": datetime.datetime.today()}  # Noncompliant
    else:
        return {"success": datetime.datetime.now().isoformat()}  # Compliant


def multiple_returns_lambda_handler(event, context):
    if event.get("format") == "raw":
        return datetime.datetime.now()  # Noncompliant
    elif event.get("format") == "iso":
        return datetime.datetime.now().isoformat()  # Compliant
    elif event.get("a") == "b":
        return
    else:
        return {"timestamp": datetime.datetime.now()}  # Noncompliant


# Compliant Lambda handlers - returning JSON serializable values
def compliant_lambda_handler(event, context):
    set_to_list = {1, 2, 3, 4, 5}
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "data": list(set_to_list),
        "nested": {
            "created_at": datetime.datetime.now().isoformat(),
            "info": "test",
        },
        "string": "test",
        "number": 42,
        "float": 3.14,
        "boolean": True,
        "null": None,
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "json": json.dumps({"test":{1,2}}),
        "unknown": unknown("test"),
        "other": Serializable(3)
    }


def compliant_custom_object_lambda_handler(event, context):
    alice = User("Alice", 30)
    return {
        "user": dataclasses.asdict(alice),
        "dict": alice.__dict__,
        "to_dict": alice.to_dict(),
    }


def return_string_lambda_handler(event, context):
    return "Simple string response"


# Edge cases
def no_return_lambda_handler(event, context):
    print("Processing...")
    # No explicit return - implicitly returns None


def complex_compliant_lambda_handler(event, context):
    users = [User("Alice", 30), User("Bob", 25)]
    return {
        "users": [dataclasses.asdict(user) for user in users],  # Compliant
        "metadata": {
            "created": datetime.datetime.now().isoformat(),  # Compliant
            "tags": list({"urgent", "important"}),  # Compliant
        },
    }


# Non-Lambda functions (should not trigger the rule)
def regular_function_set():
    return {1, 2, 3}
