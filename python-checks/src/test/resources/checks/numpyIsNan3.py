from numpy import nan


def foo(x):
    if x == nan: print(1)  # Noncompliant
    #  ^^^^^^^^
    if nan == x: print(1)  # Noncompliant
    if x != nan: print(1)  # Noncompliant
    if nan != x: print(1)  # Noncompliant
    if zz.nan != x: print(1)
    if x == zz.nan: print(1)
    if isnan(x): print(1)
    if x == zeros(42): print(1)
    if nan < x: print(1)
