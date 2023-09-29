def non_compliant_1(xx):
    import pandas as pd

    hello = foo()
    hello.values

    df = pd.DataFrame({
        'X': ['A', 'B', 'A', 'C'],
        'Y': [10, 7, 12, 5]
    })

    _ = df.values  # Noncompliant
    #      ^^^^^^
    _ = xx.values

    _ = pd.DataFrame({
        'X': ['A', 'B', 'A', 'C'],
        'Y': [10, 7, 12, 5]
    }).values  # Noncompliant
    #  ^^^^^^


def non_compliant_1(tt):
    import pandas as xx

    df = pd.DataFrame({
        'X': ['A', 'B', 'A', 'C'],
        'Y': [10, 7, 12, 5]
    })

    _ = df.values

    _ = xx.DataFrame({
        'X': ['A', 'B', 'A', 'C'],
        'Y': [10, 7, 12, 5]
    }).values  # Noncompliant
    #  ^^^^^^


def non_compliant_2():
    from pandas import DataFrame

    df = DataFrame({
        'X': ['A', 'B', 'A', 'C'],
        'Y': [10, 7, 12, 5]
    })

    _ = df.values  # Noncompliant
    #      ^^^^^^


def compliant_1():
    import pandas as pd

    df = pd.DataFrame({
        'X': ['A', 'B', 'A', 'C'],
        'Y': [10, 7, 12, 5]
    })

    _ = df.to_numpy()

    {1, 2, 3}.values


def foo():
    return 0


def compliant_2():
    from pandas import DataFrame
    df = DataFrame({
        'X': ['A', 'B', 'A', 'C'],
        'Y': [10, 7, 12, 5]
    })

    _ = df.to_numpy()
