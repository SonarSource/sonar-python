def non_compliant_1():
    import pandas as pd

    df = pd.DataFrame({'A': [3, 2, 1], 'B': ['x', 'y', 'z']})
    df2 = pd.DataFrame({'A': [6, 5, 4], 'C': ['a', 'b', 'c']})

    new_index = ['Firefox', 'Chrome', 'Safari']

    _ = df.drop(columns='A', inplace=True)  # Noncompliant
    #                        ^^^^^^^^^^^^

    _ = df.dropna(inplace=True)  # Noncompliant
    #             ^^^^^^^^^^^^

    _ = df.drop_duplicates(inplace=True)  # Noncompliant
    #                      ^^^^^^^^^^^^

    _ = df.sort_values(by=['A'], inplace=True)  # Noncompliant
    #                            ^^^^^^^^^^^^

    _ = df.sort_index(inplace=True)  # Noncompliant
    #                 ^^^^^^^^^^^^

    _ = df.eval("C = str(A) + B", inplace=True)  # Noncompliant
    #                             ^^^^^^^^^^^^

    _ = df.query('A > B', inplace=True)  # Noncompliant
    #                     ^^^^^^^^^^^^


def non_compliant_2():
    from pandas import DataFrame

    df = DataFrame({'A': [3, 2, 1], 'B': ['x', 'y', 'z']})
    df2 = DataFrame({'A': [6, 5, 4], 'C': ['a', 'b', 'c']})

    new_index = ['Firefox', 'Chrome', 'Safari']

    _ = df.drop(columns='A', inplace=True)  # Noncompliant
    #                        ^^^^^^^^^^^^

    _ = df.dropna(inplace=True)  # Noncompliant
    #             ^^^^^^^^^^^^

    _ = df.drop_duplicates(inplace=True)  # Noncompliant
    #                      ^^^^^^^^^^^^

    _ = df.sort_values(by=['A'], inplace=True)  # Noncompliant
    #                            ^^^^^^^^^^^^

    _ = df.sort_index(inplace=True)  # Noncompliant
    #                 ^^^^^^^^^^^^

    _ = df.eval("C = str(A) + B", inplace=True)  # Noncompliant
    #                             ^^^^^^^^^^^^

    _ = df.query('A > B', inplace=True)  # Noncompliant
    #                     ^^^^^^^^^^^^


def compliant(xx):
    import pandas as pd
    df = pd.DataFrame({'A': [3, 2, 1], 'B': ['x', 'y', 'z']})

    df2 = pd.DataFrame({'A': [6, 5, 4], 'C': ['a', 'b', 'c']})

    new_index = ['Firefox', 'Chrome', 'Safari']

    _ = df.drop(columns='A')

    _ = df.dropna(inplace=False)

    _ = df.drop_duplicates()

    _ = df.sort_values(by=['A'])

    _ = df.sort_index()

    _ = df.eval("C = str(A) + B")

    _ = df.query('A > B')

    _ = xx.drop(columns='A', inplace=True)

    _ = xx.dropna(inplace=False)

    _ = xx.drop_duplicates()

    _ = xx.sort_values(by=['A'])

    _ = xx.sort_index()

    _ = xx.eval("C = str(A) + B")

    _ = xx.query('A > B')
