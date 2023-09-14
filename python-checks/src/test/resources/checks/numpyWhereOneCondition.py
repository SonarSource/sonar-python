def import_1():
    import numpy as np
    def bigger_than_two():
        arr = np.array([1, 2, 3, 4])
        result = np.where(arr > 2)  # Noncompliant {{Use "np.nonzero" when only the condition parameter is provided to "np.where".}}
        #        ^^^^^^^^^^^^^^^^^

        print(0)

    def bigger_than_one():
        arr = np.array([1, 2, 3, 4])
        result = np.where(arr > 1, arr + 1, arr)
        indices = np.nonzero(arr > 1)

    # def bigger_than_one_ref():
    #     arr = np.array([1, 2, 3, 4])
    #     args = (arr > 1, arr + 1, arr)
    #     result = np.where(*args)
    #     indices = np.nonzero(arr > 1)

    def bigger_than_x():
        arr = np.array([1, 2, 3, 4])
        result = xx.where(arr > 2)
        print(0)

    def bigger_than_one():
        arr = np.array([1, 2, 3, 4])
        result = np.where(arr > 1, arr + 1)  # Compliant: There are two arguments, so we don't correct the rule. This however is an error.
        indices = np.nonzero(arr > 1)


def import_2():
    import numpy as xx

    def bigger_than_two():
        arr = xx.array([1, 2, 3, 4])
        result = xx.where(arr > 2)  # Noncompliant
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


def import_3():
    from numpy import where, nonzero, array

    def bigger_than_two():
        arr = array([1, 2, 3, 4])
        result = where(arr > 2)  # Noncompliant {{Use "np.nonzero" when only the condition parameter is provided to "np.where".}}
        #        ^^^^^^^^^^^^^^
        print(0)

    def bigger_than_one():
        arr = array([1, 2, 3, 4])
        result = where(arr > 1, arr + 1, arr)
        indices = nonzero(arr > 1)

    def bigger_than_x():
        arr = array([1, 2, 3, 4])
        result = xx.where(arr > 2)
        print(0)
