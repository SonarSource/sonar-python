import numpy as xx


def bigger_than_two():
    arr = xx.array([1, 2, 3, 4])
    result = xx.where(arr > 2)  # Noncompliant: only the condition parameter is provided to the np.where function.
    #        ^^^^^^^^^^^^^^^^^
    print(0)


def bigger_than_one():
    arr = xx.array([1, 2, 3, 4])
    result = xx.where(arr > 1, arr + 1, arr)
    indices = xx.nonzero(arr > 1)


def bigger_than_x():
    arr = xx.array([1, 2, 3, 4])
    result = np.where(arr > 2)
    print(0)
