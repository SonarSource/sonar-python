import pandas as pd
from pandas import DataFrame


def non_compliant(df: pd.DataFrame, df2: DataFrame):

    DataFrame().set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].mean().head()  # Noncompliant {{Refactor this long chain of instructions with pandas.pipe}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    df.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].mean().head()  # FN see SONARPY-1503

    df2.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"]["test"].mean().head()  # Noncompliant {{Refactor this long chain of instructions with pandas.pipe}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    pd.read_csv("some_csv.csv").filter(like='joe', axis=0).groupby("team")["salary"]["test"].mean().head() # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def compliant(df: pd.DataFrame, my_function, something, df2: DataFrame):

    df2.set_index("name").filter(like='joe', axis=0).mean().head()

    pd.read_csv("some_csv.csv").filter(like='joe', axis=0).groupby("team")["salary"]["test"].head()

    df.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].mean()

    DataFrame().set_index("name").pipe(my_function).filter(like='joe', axis=0).groupby("team")["salary"].mean()

    something.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].mean().head()
