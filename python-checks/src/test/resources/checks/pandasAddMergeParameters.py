def non_compliant_merge_1():
    import pandas as pd

    age_df = pd.DataFrame({"user_id": [1, 2, 4], "age": [42, 45, 35]})
    name_df = pd.DataFrame({"user_id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"]})

    _ = pd.merge(age_df, name_df, on="user_id")  # Noncompliant {{The 'how' and 'validate' parameters of the merge should be specified.}}

    _ = pd.merge(age_df, name_df)  # Noncompliant {{The 'how', 'on' and 'validate' parameters of the merge should be specified.}}

    _ = pd.merge(age_df, name_df, how="right")  # Noncompliant {{The 'on' and 'validate' parameters of the merge should be specified.}}

    _ = pd.merge(age_df, name_df, validate="1:1")  # Noncompliant {{The 'how' and 'on' parameters of the merge should be specified.}}

    _ = pd.merge(age_df, name_df, how="right", on="user_id")  # Noncompliant {{The 'validate' parameter of the merge should be specified.}}

    _ = pd.merge(age_df, name_df, on="user_id", validate="1:1")  # Noncompliant {{The 'how' parameter of the merge should be specified.}}

    _ = pd.merge(age_df, name_df, how="cross",  # Noncompliant {{The 'on' parameter of the merge should be specified.}}
                 validate="1:1")
    _ = age_df.merge(name_df)  # FN Noncompliant {{The 'how', 'on' and 'validate' parameters of the merge should be specified.}}

    _ = age_df.merge(name_df, on="user_id")  # FN Noncompliant {{The 'how' and 'validate' parameters of the merge should be specified.}}

    _ = age_df.merge(name_df, how="right")  # FN Noncompliant {{The 'on' and 'validate' parameters of the merge should be specified.}}

    _ = age_df.merge(name_df, validate="1:1")  # FN Noncompliant {{The 'haw' and 'on' parameters of the merge should be specified.}}

    _ = age_df.merge(name_df, how="right", on="user_id")  # FN Noncompliant {{The 'validate' parameter of the merge should be specified.}}

    _ = age_df.merge(name_df, on="user_id", validate="1:1")  # FN Noncompliant {{The 'how' parameter of the merge should be specified.}}

    _ = age_df.merge(name_df, how="cross", validate="1:1")  # FN Noncompliant {{The 'validate' parameter of the merge should be specified.}}


def non_compliant_2(xx):
    from pandas import DataFrame, merge
    age_df = DataFrame({"user_id": [1, 2, 4], "age": [42, 45, 35]})
    name_df = DataFrame({"user_id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"]})

    _ = age_df.merge(name_df)  # FN Noncompliant {{The 'how', 'on' and 'validate' parameters of the merge should be specified.}}

    _ = merge(age_df, name_df)  # Noncompliant {{The 'how', 'on' and 'validate' parameters of the merge should be specified.}}

    _ = merge(age_df, name_df, on="user_id")  # Noncompliant {{The 'how' and 'validate' parameters of the merge should be specified.}}

    _ = merge(age_df, name_df, how="right")  # Noncompliant {{The 'on' and 'validate' parameters of the merge should be specified.}}

    _ = merge(age_df, name_df, validate="1:1")  # Noncompliant {{The 'how' and 'on' parameters of the merge should be specified.}}

    _ = merge(age_df, name_df, how="right", on="user_id")  # Noncompliant {{The 'validate' parameter of the merge should be specified.}}

    _ = merge(age_df, name_df, on="user_id", validate="1:1")  # Noncompliant {{The 'how' parameter of the merge should be specified.}}

    _ = merge(age_df, name_df, how="cross", validate="1:1")  # Noncompliant {{The 'on' parameter of the merge should be specified.}}


def compliant_1(xx):
    import pandas as pd

    age_df = pd.DataFrame({"user_id": [1, 2, 4], "age": [42, 45, 35]})
    name_df = pd.DataFrame({"user_id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"]})

    _ = pd.merge(name_df, on="user_id", how="right", validate="1:1")

    _ = age_df.merge(name_df, on="user_id", how="right", validate="1:1")
