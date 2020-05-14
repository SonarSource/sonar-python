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
