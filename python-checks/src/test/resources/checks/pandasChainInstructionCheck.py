import pandas as pd
from pandas import DataFrame


def non_compliant(df: pd.DataFrame, df2: DataFrame):

    df2.set_index("name").T.filter(like='joe', axis=0)[1].add(10).mean().round().to_parquet()  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    DataFrame().set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].add(10).mean().round().to_parquet()  # Noncompliant {{Refactor this long chain of instructions with pandas.pipe}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    df.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].add(10).mean().round().to_parquet()  # FN see SONARPY-1503

    df2.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"]["test"].add(10).mean().round().to_parquet()  # Noncompliant {{Refactor this long chain of instructions with pandas.pipe}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    pd.read_csv("some_csv.csv").filter(like='joe', axis=0).groupby("team")["salary"]["test"].add(10).mean().round().to_parquet()  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#   Here we do not raise an issue only because we do not support subscription with Name
#   If support is added for such case we would encounter FPs when the subscription with Name is at the beginning of the chain 
    pd.read_csv("some_csv.csv").filter(like='joe', axis=0).add(10).groupby("team")["salary"]["test"].axes[1].unique().to_json()  # FN

#   Here we should not raise an issue as the chain is done mainly on an Index object which does not have a pipe method
    pd.read_csv("some_csv.csv").axes[1].join(pd.Index([4, 5, 6])).T.repeat([1,2]).drop_duplicates().insert(1, 42).sort_values()

def compliant(df: pd.DataFrame, my_function, something, df2: DataFrame):

    df2.set_index("name").T.filter(like='joe', axis=0)[1].add(10).mean().to_html()

    (df2.set_index("name").T.filter(like='joe', axis=0))[1].add(10).mean().round().to_html()

    df2.set_index("name").filter(like='joe', axis=0).mean().head()

    pd.read_csv("some_csv.csv").filter(like='joe', axis=0).groupby("team")["salary"]["test"].head()

    df.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].mean()

    DataFrame().set_index("name").pipe(my_function).filter(like='joe', axis=0).groupby("team")["salary"].add(10).round().mean().to_json()

    something.set_index("name").filter(like='joe', axis=0).groupby("team")["salary"].add(10).round().mean().to_parquet()

