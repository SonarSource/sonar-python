import pandas as pd


def non_compliant(df: pd.DataFrame):
    df.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].mean().head()  # Noncompliant {{Refactor this long chain of instructions with pandas.pipe}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    df.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"]["test"].mean().head()  # Noncompliant {{Refactor this long chain of instructions with pandas.pipe}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    pd.read_csv("some_csv.csv").filter(like='joe', axis=0).groupby("team")["salary"]["test"].mean().head() # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def compliant(df: pd.DataFrame, my_function, something):
    df.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].mean()

    df.set_index("name").pipe(my_function).filter(like='joe', axis=0).groupby("team")["salary"].mean()

    something.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].mean().head()
