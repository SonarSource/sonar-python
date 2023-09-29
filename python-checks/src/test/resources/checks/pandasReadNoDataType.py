def non_compliant_1(xx):
    import pandas as pd

    _ = pd.read_csv("my_file.csv")  # Noncompliant {{Provide the 'dtype' parameter when calling 'pandas.read_csv'.}}
    #      ^^^^^^^^

    _ = pd.read_table("my_file.csv")  # Noncompliant {{Provide the 'dtype' parameter when calling 'pandas.read_table'.}}
    #      ^^^^^^^^^^

    _ = pd.read_csv(xx)  # Noncompliant {{Provide the 'dtype' parameter when calling 'pandas.read_csv'.}}
    #      ^^^^^^^^


def non_compliant_2():
    from pandas import read_csv, read_table

    _ = read_csv("my_file.csv")  # Noncompliant {{Provide the 'dtype' parameter when calling 'pandas.read_csv'.}}
    #   ^^^^^^^^
    _ = read_table("my_file.csv")  # Noncompliant {{Provide the 'dtype' parameter when calling 'pandas.read_table'.}}
    #   ^^^^^^^^^^


def compliant_1(xx):
    import pandas as pd
    import numpy as np

    _ = pd.read_csv(
        "my_file.csv",
        dtype={'name': 'str', 'age': 'int'})

    _ = pd.read_table(
        "my_file.csv",
        dtype={'name': 'str', 'age': 'int'})

    _ = xx.read_csv("my_file.csv")

    _ = xx.read_table("my_file.csv")

    _ = np.array([1, 2, 3])


def compliant_2():
    from pandas import read_csv, read_table

    _ = read_csv(
        "my_file.csv",
        dtype={'name': 'str', 'age': 'int'})

    _ = read_table(
        "my_file.csv",
        dtype={'name': 'str', 'age': 'int'})
