import numpy as np


def foo(x):
    if x == np.nan: print(1)  # Noncompliant
    #  ^^^^^^^^^^^
    if np.nan == x: print(1)  # Noncompliant
    if x != np.nan: print(1)  # Noncompliant
    if np.nan != x: print(1)  # Noncompliant
    if zz.nan != x: print(1)
    if np.isnan(x): print(1)
    if x == np.zeros(42): print(1)
    if np.nan < x: print(1)  # Compliant
