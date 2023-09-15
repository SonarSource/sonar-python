def import_1():
    import numpy as np
    arr = np.array([1, 2, 3, 4])

    empty = np.where()

    empty = np.where(yy)  # Noncompliant
    #       ^^^^^^^^^^^^

    result = np.where(arr > 2)  # Noncompliant {{Use "np.nonzero" when only the condition parameter is provided to "np.where".}}
    #        ^^^^^^^^^^^^^^^^^

    args = (arr > 1, arr + 1, arr)
    result = np.where(*args)

    args = (arr > 1,)
    result = np.where(*args)  # Compliant: This is a FN. The check hasn't been yet implemented for detecting this.

    indices = np.nonzero(arr > 1)

    result = np.where(arr > 1, arr + 1, arr)

    result = xx.where(arr > 2)

    result = np.where(arr > 1, arr + 1)  # Compliant: There are two arguments, so we don't correct the rule. This however is an error.

    param = np.where(x=arr + 1)

    param = np.where(condition=arr > 3)  # Noncompliant


def import_2():
    import numpy as xx
    arr = xx.array([1, 2, 3, 4])

    result = xx.where(arr > 2)  # Noncompliant
    #        ^^^^^^^^^^^^^^^^^

    result = xx.where(arr > 1, arr + 1, arr)


def import_3():
    from numpy import where, nonzero, array
    arr = array([1, 2, 3, 4])

    result = where(arr > 2)  # Noncompliant {{Use "np.nonzero" when only the condition parameter is provided to "np.where".}}
    #        ^^^^^^^^^^^^^^

    result = where(arr > 1, arr + 1, arr)
    indices = nonzero(arr > 1)
