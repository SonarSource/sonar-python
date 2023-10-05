import pandas as pd
from pandas import DataFrame


def non_compliant(df: pd.DataFrame, df2: DataFrame):

    df2.set_index("name").T.filter(like='joe', axis=0)[1].mean().head()  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    DataFrame().set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].mean().head()  # Noncompliant {{Refactor this long chain of instructions with pandas.pipe}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    df.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].mean().head()  # FN see SONARPY-1503

    df2.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"]["test"].mean().head()  # Noncompliant {{Refactor this long chain of instructions with pandas.pipe}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    pd.read_csv("some_csv.csv").filter(like='joe', axis=0).groupby("team")["salary"]["test"].mean().head() # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#   Here we do not raise an issue only because we do not support subscription with Name
#   If support is added for such case we would encounter FPs when the subscription with Name is at the beginning of the chain 
    pd.read_csv("some_csv.csv").filter(like='joe', axis=0).groupby("team")["salary"]["test"].axes[1].unique() # FN

#   Here we should not raise an issue as the chain is done mainly on an Index object which does not have a pipe method
    pd.read_csv("some_csv.csv").axes[1].join(pd.Index([4, 5, 6])).repeat([1,2]).drop_duplicates().insert(1, 42)

def compliant(df: pd.DataFrame, my_function, something, df2: DataFrame):

    df2.set_index("name").T.filter(like='joe', axis=0)[1].mean()

    (df2.set_index("name").T.filter(like='joe', axis=0))[1].mean()

    df2.set_index("name").filter(like='joe', axis=0).mean().head()

    pd.read_csv("some_csv.csv").filter(like='joe', axis=0).groupby("team")["salary"]["test"].head()

    df.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].mean()

    DataFrame().set_index("name").pipe(my_function).filter(like='joe', axis=0).groupby("team")["salary"].mean()

    something.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].mean().head()

