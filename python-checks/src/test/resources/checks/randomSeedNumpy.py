import numpy as np

def failure():
    gen = np.random.default_rng()  # Noncompliant {{Provide a seed for this random generator.}}
    #     ^^^^^^^^^^^^^^^^^^^^^

    gen = np.random.SeedSequence()  # Noncompliant
    #     ^^^^^^^^^^^^^^^^^^^^^^

    gen = np.random.SeedSequence(entropy=None)  # Noncompliant
    #     ^^^^^^^^^^^^^^^^^^^^^^
    gen = np.random.SeedSequence(spawn_key=[123])  # Noncompliant
    #     ^^^^^^^^^^^^^^^^^^^^^^

    from numpy.random import SFC64, MT19937, PCG64, PCG64DXSM, Philox

    gen = SFC64()  # Noncompliant
    #     ^^^^^
    gen = MT19937(seed=None)  # Noncompliant
    #     ^^^^^^^
    gen = PCG64()  # Noncompliant
    #     ^^^^^

    gen = PCG64DXSM()  # Noncompliant
    #     ^^^^^^^^^
    a = None
    gen = Philox(a)  # Noncompliant
    #     ^^^^^^

    gen = np.random.seed()  # Noncompliant
    #     ^^^^^^^^^^^^^^
    gen = np.seed(None)  # Noncompliant
    #     ^^^^^^^


def success():
    gen = np.random.default_rng(42)

    gen = np.random.SeedSequence(42)
    gen = np.random.SeedSequence(entropy=[123])
    gen = np.random.SeedSequence(123, spawn_key=[123])

    def test_default_value(seed=None):
        np.random.SeedSequence(seed)


    def test_seed(seed):
        np.random.SeedSequence(seed)

    from numpy.random import SFC64, MT199937, PCG64, PCG64DXSM, Philox

    gen = SFC64(42)
    gen = MT199937(seed=10)
    gen = PCG64(0)

    gen = PCG64DXSM(2)
    a = 8
    gen = Philox(a)

    gen = np.random.seed(13)
    gen = np.seed(seed=9)
