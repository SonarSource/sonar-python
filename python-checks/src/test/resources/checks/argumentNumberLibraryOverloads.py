import os
from os import getenv as environment

os.getenv()  # Overloads disagree on the expected argument count.
os.getenv("HOME")
os.getenv("HOME", None)
os.getenv("HOME", "fallback")
environment()  # Keep conflicting corrections conservative.
environment("HOME", unexpected=True)  # Noncompliant
round()  # Overloads disagree on the correction; preserve consensus behavior.
round(1.5)
round(1.5, 1)
round(1.5, 1, 2)  # Noncompliant


def source_overload_declarations_remain_conservative():
    from typing import overload

    @overload
    def declared(value): ...

    declared()


def sorting_and_tracebacks():
    import traceback
    values = [3, 1, 2]
    values.sort(key=lambda value: value, reverse=True)
    values.sort(cmp=lambda left, right: left - right)  # Noncompliant {{Remove this unexpected named argument 'cmp'.}}
    sorted(values, key=lambda value: value)
    sorted(values, lambda left, right: left - right)  # Noncompliant
    traceback.format_exception(ValueError("failure"))
    traceback.format_exception()  # Noncompliant {{Add 1 missing arguments; 'format_exception' expects at least 1 positional arguments.}}


def mixed_source_and_library_overloads(condition):
    def local(key): ...
    candidate = local if condition else os.getenv
    candidate("HOME")
    candidate("HOME", unexpected=True)  # Noncompliant {{Remove this unexpected named argument 'unexpected'.}}

    def accepts_keywords(key, **kwargs): ...
    candidate = accepts_keywords if condition else os.getenv
    candidate("HOME", unexpected=True)
