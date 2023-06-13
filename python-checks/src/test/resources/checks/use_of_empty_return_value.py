def bar(xs): ...


def assignement_statement():
    ls = [1, 2, 3]
    result = ls.append(42)  # Noncompliant {{Remove this use of the output from "append"; "append" doesn’t return anything.}}
#            ^^^^^^^^^^^^^
    n = ls.count(1)  # OK, count returns a integer
    return result, n


def passed_as_argument():
    ls = [1, 2, 3]
    bar(ls.sort())  # Noncompliant {{Remove this use of the output from "sort"; "sort" doesn’t return anything.}}
#       ^^^^^^^^^
    bar(xs=ls.sort())  # Noncompliant
#          ^^^^^^^^^
    bar(*ls.sort())  # This should be raised by S5633
    bar(**ls.sort())  # This should be raised by S5633
    bar(ls.copy())
    x = bar()  # OK, we don't know the type of bar


ls = [1, 2, 3]
result = ls.append(42)  # FN - no type inference outside of functions


def win32():
    import win32pdh
    # We should not raise issues as the stubs of win32 are inaccurate
    path = win32pdh.MakeCounterPath((None, None, None, None, -1, None)) # OK
    hc = win32pdh.AddCounter(None, path)
