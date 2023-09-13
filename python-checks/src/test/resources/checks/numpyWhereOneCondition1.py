import numpy as np


def bigger_than_two():
    arr = np.array([1, 2, 3, 4])
    result = np.where(arr > 2)  # Noncompliant: only the condition parameter is provided to the np.where function.
    #        ^^^^^^^^^^^^^^^^^
    print(0)


def bigger_than_one():
    arr = np.array([1, 2, 3, 4])
    result = np.where(arr > 1, arr + 1, arr)
    indices = np.nonzero(arr > 1)


def bigger_than_x():
    arr = np.array([1, 2, 3, 4])
    result = xx.where(arr > 2)
    print(0)


def bigger_than_one():
    arr = np.array([1, 2, 3, 4])
    result = np.where(arr > 1, arr + 1)  # Compliant: There are two arguments, so we don't correct the rule. This however is an error.
    indices = np.nonzero(arr > 1)
