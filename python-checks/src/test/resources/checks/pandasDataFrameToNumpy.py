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


def dataframe_from_read_csv():
    import pandas as pd

    my_df = pd.read_csv("some_csv.csv")
    my_df.values # Noncompliant


def dataframe_from_pandas_core():
    from pandas.core.frame import DataFrame

    DataFrame({"column": [1]}).values # Noncompliant
    #                          ^^^^^^


def fn_on_parameter_type_hint():
    import pandas as pd

    def dataframe_param(df : pd.DataFrame):
        df.values # FN


def direct_read_csv():
    from io import StringIO
    import pandas as pd
    from pandas import read_csv as load_csv

    pd.read_csv(StringIO("1"), dtype=int, header=None).values[0][0]  # Noncompliant
    load_csv("some_csv.csv").values  # Noncompliant
    (pd.read_csv("some_csv.csv")).values  # Noncompliant


def unrelated_read_csv():
    def read_csv(path):
        return object()

    read_csv("some_csv.csv").values


def shadowed_pandas(pd):
    pd.read_csv("some_csv.csv").values


def aliased_dataframe():
    import pandas as pd

    dataframe = pd.read_csv("some_csv.csv")
    alias = dataframe
    alias.values  # Noncompliant


def cyclic_aliases():
    first = second
    second = first
    first.values


def self_alias():
    value = value
    value.values
