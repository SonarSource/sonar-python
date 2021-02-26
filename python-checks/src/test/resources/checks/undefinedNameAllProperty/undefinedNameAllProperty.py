from math import exp
from unresolved import Unresolved
from .same_package import same_package_function

var = "unknown"
#     ^^^^^^^^^> {{Assigned here.}}
not_a_string = 42

__all__ = [
    'LocalClass',
    '',  # Noncompliant
    Unresolved,  # Covered by S3827
    undefined_variable,  # Covered by S3827
    'Unresolved',
    'Unresolved.something', # Noncompliant
    'NeverDefined',   # Noncompliant {{Change or remove this string; "NeverDefined" is not defined.}}
    'exp',
    'method',   # Noncompliant
    'HidingClass.method', # Noncompliant
    'hidden_variable',   # Noncompliant
    'NestedClass',  # Noncompliant
    exp.__name__,
    "unicode", # OK (builtin symbol)
    not_a_string, # Out of scope
    f'interpolated{var}', # Out of scope
    var  # Noncompliant {{Change or remove this string; "unknown" is not defined.}}
#   ^^^
]
__all__ = (
  'LocalClass',
  'NeverDefined' # Noncompliant
)

__all__ = 42 # Out of scope
__all__, b = ("a", "b") # Out of scope
a.__all__ = ("a", "b") # Out of scope
c = __all__

class LocalClass:
    pass

def local_function():
    pass

class HidingClass:
    def method(self):
        hidden_variable = 42

    class NestedClass:
        pass
