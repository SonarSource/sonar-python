from typing import Callable, Any

def lambda_assignment_check(param):
    a = lambda x: x # Noncompliant {{Define function instead of this lambda assignment statement.}}
    #   ^^^^^^
    a, b = lambda x: x, 10 # Compliant: If a developer uses assignment expressions like this, it might be intentional
    a = b = lambda x: x # Noncompliant
    #       ^^^^^^
    (b := lambda x: x)(10) # Compliant: If a developer uses assignment expressions like this, it might be intentional
    a: Callable[[Any], Any] = lambda x: x # Noncompliant {{Define function instead of this lambda assignment statement.}}

    a: Callable[[Any], Any] # For coverage

def assignment_in_default_parameter(param = lambda x: x): # Compliant: Lambda default parameters are likely extremly simple functions / not reused
    pass

class LambdaAssignmentInFields:
    static_field = lambda x: x # Noncompliant
    #              ^^^^^^

    def __init__(self):
        self.field = lambda x: x # Noncompliant
