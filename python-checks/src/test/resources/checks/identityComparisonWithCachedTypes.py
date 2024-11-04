def literal_comparison(param):
    3000 is param # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #    ^^
    param is 0.1 # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    0.3e5 is param # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is b'8259' # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    b'5363' is param # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #       ^^
    'str' is param # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is frozenset([42, 58]) # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    frozenset(param) is param # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #                ^^
    (4, 5) is param # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #      ^^

    param is -1 # Noncompliant
    param is 1 + 1 # Noncompliant

def literal_comparison_compliant(param):
    param is param # ok from the point of this rule
    param is someUnknownFunction(42) # ok
    param == param # ok
    param == 2000 # ok
    0.1 == param # ok
    param == b'01'
    "str" == param
    (4, 5) == param

def literal_comparison_with_not_token(param):
    3000 is not param # Noncompliant {{Replace this "is not" operator with "!="; identity operator is not reliable here.}}
    #    ^^^^^^
    param is not (1, 2, 3) # Noncompliant {{Replace this "is not" operator with "!="; identity operator is not reliable here.}}
    #     ^^^^^^

def literal_comparison_inside_more_interesting_expressions(param):
  x = 42 if param is not (2, 3) else 58 # Noncompliant {{Replace this "is not" operator with "!="; identity operator is not reliable here.}}
  #               ^^^^^^
  y = [x * x for x in range(10) if x is not 5] # Noncompliant {{Replace this "is not" operator with "!="; identity operator is not reliable here.}}
  #                                  ^^^^^^

def reference_escape_two_method_variables(x, y):
  x is y # ok, both variables could have come from anywhere, they could actually be identical

def integer_assignment():
  """This is actually 'true', but it wouldn't hurt to use == here either."""
  # Even though it adheres to the spec,
  # this specific test was not in the specification, it could be argued that it's actually valid, just very uncommon.
  x = 42
  y = x
  x is y # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
  # ^^

def reference_escape_assignment_through_unknown_function(f):
  x = 42
  y = f(x)
  x is y # ok, `f` could be identity, `x` could have ended up in `y`. It wouldn't be too bad if it was marked, though.

def reference_escape_assignment_into_array(self):
    a = []
    x = (1, 2, 3, 4, 5)
    a[0] = x
    assert_equal(a[0], x)
    assert_(a[0] is x)

# Corner cases for code coverage
def coverage():
  ambiguous = "" if 42 * 42 < 1700 else (lambda x: x * x)
  ambiguous is unknown
  f(x) is g(y)
  f.g(h) is c.d(e)
  f is g


symbollessName is somethingElse # for `null`-Symbols on names


# rest is mutably borrowed from `expected-issues/python/src/RSPEC_5795`.

def literal_comparison(param):
    param is 2000  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is b"a"  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is 3.0  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is "test"  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is u"test"  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is (1, 2, 3)  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is not 2000  # Noncompliant {{Replace this "is not" operator with "!="; identity operator is not reliable here.}}
    #     ^^^^^^
    2000 is param  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #    ^^


def functions_returning_cached_types(param):
    param is int("1000")  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is bytes(1)  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is float("1.0")  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is str(1000)  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is tuple([1, 2, 3])  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is frozenset([1, 2, 3])  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is hash("a")  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^


def variables(param):
    var = 1
    param is var  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    SENTINEL = (0, 1)
    param is SENTINEL  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    SENTINEL is param  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #        ^^

def compliant_bool_and_none(param):
    param is True # ok
    param is False # ok
    param is bool(1)
    param is None

def noncompliant_even_if_it_works_with_cpython(param):
    param is ()  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is tuple()  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is ""  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^
    param is 1  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #     ^^

def default_param(param=(0, 1)):
    print(param is (0, 1))  # Noncompliant {{Replace this "is" operator with "=="; identity operator is not reliable here.}}
    #           ^^

def comparison_to_none():
    "str" is not None
    "str" is None
    123 is None
    None is "str"

def potential_null_symbols():
    type(arr) is tuple
    some_thing is other_thing

def comparison_to_class(arg):
    import typing
    arg is not typing.Tuple

def resolved_integer_variables(arg):
    import logging
    import subprocess

    arg is logging.INFO # Noncompliant
    arg is subprocess.STDOUT # Noncompliant
