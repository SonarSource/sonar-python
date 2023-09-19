from typing import Any


def import_1():
    import numpy as np
    from typing import Any

    gen_1 = (x * 2 for x in range(5))
    arr = np.array(gen_1)  # Noncompliant {{Pass a list to "np.array" instead of passing a generator.}}
    #     ^^^^^^^^^^^^^^^

    gen_2: Any = (x * 2 for x in range(5))  # This is a false negative. ReachingDefinitionsAnalysis does not find the variable when it has
    # been type annotated. Let's construct a ticket for this.
    arr = np.array(gen_2)

    gen_3 = 42
    np.array(gen_3)

    def test(xx):
        np.array(xx)
        np.array(xx, gen_1)

    arr = np.array((x ** 2 for x in range(10)), dtype=object)  # Compliant: the dtype parameter of np.array is set to object.

    arr = np.array(x ** 2 for x in range(10))  # Noncompliant {{Pass a list to "np.array" instead of passing a generator.}}
    #     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    arr = np.array([x ** 2 for x in range(10)])  # Compliant: a list of 10 elements is passed to the np.array function.

    arr = np.array((x ** 2 for x in range(10)), dtype=str)  # Noncompliant

    arr = np.array()

    gen_4 = (x * 2 for x in range(5))
    np.array(gen_4)  # Noncompliant
