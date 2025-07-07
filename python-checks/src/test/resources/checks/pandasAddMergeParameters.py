def non_compliant_merge_1():
    import pandas as pd

    age_df = pd.read_csv("age_csv.csv")
    name_df = pd.read_csv("name_csv.csv")

    _ = age_df.merge(name_df)  # Noncompliant {{Specify the "how", "on" and "validate" parameters of this merge.}}

    _ = age_df.merge(name_df, on="user_id")  # Noncompliant {{Specify the "how" and "validate" parameters of this merge.}}

    _ = age_df.merge(name_df, how="right")  # Noncompliant {{Specify the "on" and "validate" parameters of this merge.}}

    _ = age_df.merge(name_df, validate="1:1")  # Noncompliant {{Specify the "how" and "on" parameters of this merge.}}

    _ = age_df.merge(name_df, how="right", on="user_id")  # Noncompliant {{Specify the "validate" parameter of this merge.}}

    _ = age_df.merge(name_df, on="user_id", validate="1:1")  # Noncompliant {{Specify the "how" parameter of this merge.}}


    _ = pd.merge(age_df, name_df, on="user_id")  # Noncompliant {{Specify the "how" and "validate" parameters of this merge.}}

    _ = pd.merge(age_df, name_df)  # Noncompliant {{Specify the "how", "on" and "validate" parameters of this merge.}}

    _ = pd.merge(age_df, name_df, how="right")  # Noncompliant {{Specify the "on" and "validate" parameters of this merge.}}

    _ = pd.merge(age_df, name_df, validate="1:1")  # Noncompliant {{Specify the "how" and "on" parameters of this merge.}}

    _ = pd.merge(age_df, name_df, how="right", on="user_id")  # Noncompliant {{Specify the "validate" parameter of this merge.}}

    _ = pd.merge(age_df, name_df, on="user_id", validate="1:1")  # Noncompliant {{Specify the "how" parameter of this merge.}}


    _ = age_df.join(name_df)  # Noncompliant {{Specify the "how", "on" and "validate" parameters of this join.}}

    _ = age_df.join(name_df, on="user_id")  # Noncompliant {{Specify the "how" and "validate" parameters of this join.}}

    _ = age_df.join(name_df, how="right")  # Noncompliant {{Specify the "on" and "validate" parameters of this join.}}

    _ = age_df.join(name_df, validate="1:1")  # Noncompliant {{Specify the "how" and "on" parameters of this join.}}

    _ = age_df.join(name_df, how="right", on="user_id")  # Noncompliant {{Specify the "validate" parameter of this join.}}

    _ = age_df.join(name_df, on="user_id", validate="1:1")  # Noncompliant {{Specify the "how" parameter of this join.}}



def non_compliant_2():
    from pandas import DataFrame, merge
    age_df = DataFrame({"user_id": [1, 2, 4], "age": [42, 45, 35]})
    name_df = DataFrame({"user_id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"]})

    _ = age_df.merge(name_df)  # Noncompliant {{Specify the "how", "on" and "validate" parameters of this merge.}}

    _ = merge(age_df, name_df)  # Noncompliant {{Specify the "how", "on" and "validate" parameters of this merge.}}

    _ = merge(age_df, name_df, on="user_id")  # Noncompliant {{Specify the "how" and "validate" parameters of this merge.}}

    _ = merge(age_df, name_df, how="right")  # Noncompliant {{Specify the "on" and "validate" parameters of this merge.}}

    _ = merge(age_df, name_df, validate="1:1")  # Noncompliant {{Specify the "how" and "on" parameters of this merge.}}

    _ = merge(age_df, name_df, how="right", on="user_id")  # Noncompliant {{Specify the "validate" parameter of this merge.}}

    _ = merge(age_df, name_df, on="user_id", validate="1:1")  # Noncompliant {{Specify the "how" parameter of this merge.}}



def compliant_1(xx):
    import pandas as pd
    from pandas import merge

    age_df = pd.DataFrame({"user_id": [1, 2, 4], "age": [42, 45, 35]})
    name_df = pd.DataFrame({"user_id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"]})

    _ = pd.merge(name_df, on="user_id", how="right", validate="1:1")

    _ = age_df.merge(name_df, on="user_id", how="right", validate="1:1")

    _ = xx.merge(age_df, name_df)

    _ = xx.merge(name_df, on="user_id", how="right", validate="1:1")

    _ = pd.merge(age_df, name_df, "inner", "user_id", None, None, False, False, False, ('_x', '_y'), None, False, "1:1")

    _ = age_df.merge(name_df, "inner", "user_id", None, None, False, False, False, ('_x', '_y'), None, False, "1:1")

    _ = age_df.join(name_df, on=None, how='left', lsuffix='', rsuffix='', sort=False, validate=None)

    _ = pd.merge(age_df, name_df, left_on=col, right_on='cat', how='left', validate='m:m')

    _ = age_df.merge(name_df, left_on=col, right_on='cat', how='left', validate='m:m')

    _ = age_df.merge(name_df, left_on=col, right_on='cat', how='left', validate='m:m')

    _ = pd.merge(age_df, name_df, right_on='cat', how='left', validate='m:m')

    _ = pd.merge(age_df, name_df, how="cross",  # Cross should not provide the on argument
                 validate="1:1")

    _ = age_df.join(name_df, how="cross", validate="1:1")  # Cross should not provide the on argument

    _ = age_df.merge(name_df, how="cross", validate="1:1") # Cross should not provide the on argument

    _ = age_df.merge(name_df, right_on='cat', how='left', validate='m:m')

    _ = age_df.merge(name_df, right_on='cat', how='left', validate='m:m')

    _ = pd.merge(age_df, name_df, left_on=col, how='left', validate='m:m')

    _ = age_df.merge(name_df, left_on=col, how='left', validate='m:m')

    _ = age_df.merge(name_df, left_on=col, how='left', validate='m:m')

