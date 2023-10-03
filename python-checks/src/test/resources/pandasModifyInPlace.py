def non_compliant():
    import pandas as pd

    df = pd.DataFrame({'A': [3, 2, 1], 'B': ['x', 'y', 'z']})
    df2 = pd.DataFrame({'A': [6, 5, 4], 'C': ['a', 'b', 'c']})

    new_index = ['Firefox', 'Chrome', 'Safari']

    df.drop(columns='A', inplace=True)  # Noncompliant

    df.dropna(inplace=True)  # Noncompliant

    df.drop_duplicates(inplace=True)  # Noncompliant

    df.sort_values(by=['A'], inplace=True)  # Noncompliant

    df.sort_index(inplace=True)  # Noncompliant

    df.query('A > B', inplace=True)  # Noncompliant

    # 'transpose' actually doesn't have an 'inplace' keyword argument.
    df.transpose(inplace=True)  # Noncompliant

    df.swapaxes(inplace=True)  # Noncompliant

    df.reindex(new_index, inplace=True)  # Noncompliant

    # 'reindex_like' actually doesn't have an 'inplace' keyword argument.
    df.reindex_like(df2, inplace=True)  # Noncompliant

    # 'truncate' actually doesn't have an 'inplace' keyword argument.
    df.truncate(before="2", after="1", axis="A", inplace=True)  # Noncompliant


def compliant(xx):
    import pandas as pd
    df = pd.DataFrame({'A': [3, 2, 1], 'B': ['x', 'y', 'z']})

    df2 = pd.DataFrame({'A': [6, 5, 4], 'C': ['a', 'b', 'c']})

    new_index = ['Firefox', 'Chrome', 'Safari']

    df.drop(columns='A')

    df.dropna(inplace=False)

    df.drop_duplicates()

    df.sort_values(by=['A'])

    df.sort_index()

    df.query('A > B')

    df.transpose(inplace=False)

    df.swapaxes()

    df.reindex(new_index)

    df.reindex_like(df2)

    df.truncate(before="2", after="1", axis="A")

    xx.drop(columns='A', inplace=True)
