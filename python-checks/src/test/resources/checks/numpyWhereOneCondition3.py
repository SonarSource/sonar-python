from numpy import where, nonzero, array


def bigger_than_two():
    arr = array([1, 2, 3, 4])
    result = where(arr > 2)  # Noncompliant: only the condition parameter is provided to the np.where function.
    #        ^^^^^^^^^^^^^^^^^
    print(0)


def bigger_than_one():
    arr = array([1, 2, 3, 4])
    result = where(arr > 1, arr + 1, arr)
    indices = nonzero(arr > 1)


def bigger_than_x():
    arr = array([1, 2, 3, 4])
    result = xx.where(arr > 2)
    print(0)
