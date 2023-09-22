def foo1(x):
    import numpy as np
    if x == np.nan: print(1)  # Noncompliant {{Don't perform an equality/inequality check against "numpy.nan".}}
    #  ^^^^^^^^^^^
    if np.nan == x: print(1)  # Noncompliant
    if x != np.nan: print(1)  # Noncompliant
    if np.nan != x: print(1)  # Noncompliant
    if zz.nan != x: print(1)
    if x == zz.nan: print(1)
    if np.isnan(x): print(1)
    if x == np.zeros(42): print(1)
    if np.nan < x: print(1)
    if np.nan == np.array([1, 2, 3]): print(1)  # Noncompliant
    if np.nan == np.max(5, 3): print(1)  # Noncompliant


def foo2(x):
    import numpy as xx
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


def foo3(x):
    from numpy import nan, isnan, zeros
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
