
def noncompliant(a, b):
    import numpy as np

    np.random.randn()  # Noncompliant
#   ^^^^^^^^^^^^^^^
    np.random.beta(a, b)  # Noncompliant
#   ^^^^^^^^^^^^^^
    np.random.binomial(a, b)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^
    np.random.bytes(10)  # Noncompliant
#   ^^^^^^^^^^^^^^^
    np.random.standard_normal()  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^
    np.random.RandomState.f(a, b)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^
    np.random.RandomState.poisson()  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    RandomState(MT19937(SeedSequence(123456789)))  # Noncompliant
#   ^^^^^^^^^^^


def compliant(a, b):
    import numpy as np

    np.random.seed(42)  # Compliant: even though this is a legacy function, it is still widely used, raising an issue here would be too noisy.

    generator = np.random.default_rng(42)
    generator.standard_normal()

    generator.f(a, b)
    generator.poisson()

    from numpy.random import Generator, PCG64
    rng = Generator(PCG64())
    rng.standard_normal()
