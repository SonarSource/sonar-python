import numpy as xx


def foo(x):
    if x == xx.nan: print(1)  # Noncompliant
    #  ^^^^^^^^^^^
    if xx.nan == x: print(1)  # Noncompliant
    if x != xx.nan: print(1)  # Noncompliant
    if xx.nan != x: print(1)  # Noncompliant
    if zz.nan != x: print(1)
    if x == zz.nan: print(1)
    if xx.isnan(x): print(1)
    if x == xx.zeros(42): print(1)
    if xx.nan < x: print(1)
