def non_compliant_1():
    import pandas as pd

    df = pd.DataFrame({'A': [3, 2, 1], 'B': ['x', 'y', 'z']})
    df2 = pd.read_csv("some_csv.csv")

    _ = df.drop(columns='A', inplace=True)  # Noncompliant  {{Do not use "inplace=True" when modifying a dataframe.}}
    #                        ^^^^^^^^^^^^
    _ = df2.drop(columns='A', inplace=True)  # Noncompliant
    #                         ^^^^^^^^^^^^

    _ = df.dropna(inplace=True)  # Noncompliant
    #             ^^^^^^^^^^^^
    _ = df2.dropna(inplace=True)  # Noncompliant
    #              ^^^^^^^^^^^^

    _ = df.drop_duplicates(inplace=True)  # Noncompliant
    #                      ^^^^^^^^^^^^
    _ = df2.drop_duplicates(inplace=True)  # Noncompliant
    #                       ^^^^^^^^^^^^

    _ = df.sort_values(by=['A'], inplace=True)  # Noncompliant
    #                            ^^^^^^^^^^^^
    _ = df2.sort_values(by=['A'], inplace=True)  # Noncompliant
    #                             ^^^^^^^^^^^^

    _ = df.sort_index(inplace=True)  # Noncompliant
    #                 ^^^^^^^^^^^^
    _ = df2.sort_index(inplace=True)  # Noncompliant
    #                  ^^^^^^^^^^^^

    _ = df.eval("C = str(A) + B", inplace=True)  # Noncompliant
    #                             ^^^^^^^^^^^^
    _ = df2.eval("C = str(A) + B", inplace=True)  # Noncompliant
    #                              ^^^^^^^^^^^^

    _ = df.query('A > B', inplace=True)  # Noncompliant
    #                     ^^^^^^^^^^^^
    _ = df2.query('A > B', inplace=True)  # Noncompliant
    #                      ^^^^^^^^^^^^


def non_compliant_2():
    from pandas import DataFrame, read_csv

    df = DataFrame({'A': [3, 2, 1], 'B': ['x', 'y', 'z']})
    df2 = read_csv("some_csv.csv")

    _ = df.drop(columns='A', inplace=True)  # Noncompliant
    #                        ^^^^^^^^^^^^
    _ = df2.drop(columns='A', inplace=True)  # Noncompliant
    #                         ^^^^^^^^^^^^

    _ = df.dropna(inplace=True)  # Noncompliant
    #             ^^^^^^^^^^^^
    _ = df2.dropna(inplace=True)  # Noncompliant
    #              ^^^^^^^^^^^^

    _ = df.drop_duplicates(inplace=True)  # Noncompliant
    #                      ^^^^^^^^^^^^
    _ = df2.drop_duplicates(inplace=True)  # Noncompliant
    #                       ^^^^^^^^^^^^

    _ = df.sort_values(by=['A'], inplace=True)  # Noncompliant
    #                            ^^^^^^^^^^^^
    _ = df2.sort_values(by=['A'], inplace=True)  # Noncompliant
    #                             ^^^^^^^^^^^^

    _ = df.sort_index(inplace=True)  # Noncompliant
    #                 ^^^^^^^^^^^^
    _ = df2.sort_index(inplace=True)  # Noncompliant
    #                  ^^^^^^^^^^^^

    _ = df.eval("C = str(A) + B", inplace=True)  # Noncompliant
    #                             ^^^^^^^^^^^^
    _ = df2.eval("C = str(A) + B", inplace=True)  # Noncompliant
    #                              ^^^^^^^^^^^^

    _ = df.query('A > B', inplace=True)  # Noncompliant
    #                     ^^^^^^^^^^^^
    _ = df2.query('A > B', inplace=True)  # Noncompliant
    #                      ^^^^^^^^^^^^


def compliant(xx):
    import pandas as pd
    df = pd.DataFrame({'A': [3, 2, 1], 'B': ['x', 'y', 'z']})

    df2 = pd.read_csv("some_csv.csv")

    _ = df.drop(columns='A')
    _ = df2.drop(columns='A')

    _ = df.dropna(inplace=False)
    _ = df2.dropna(inplace=False)

    _ = df.drop_duplicates()
    _ = df2.drop_duplicates()

    _ = df.sort_values(by=['A'])
    _ = df2.sort_values(by=['A'])

    _ = df.sort_index()
    _ = df2.sort_index()

    _ = df.eval("C = str(A) + B")
    _ = df2.eval("C = str(A) + B")

    _ = df.query('A > B')
    _ = df2.query('A > B')

    _ = xx.drop(columns='A', inplace=True)

    _ = xx.dropna(inplace=False)

    _ = xx.drop_duplicates()

    _ = xx.sort_values(by=['A'])

    _ = xx.sort_index()

    _ = xx.eval("C = str(A) + B")

    _ = xx.query('A > B')
