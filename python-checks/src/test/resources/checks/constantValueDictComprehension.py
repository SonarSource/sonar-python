def case1():
    keys = ['a', 'b', 'c']
    some_v = "v"
    # Non-compliant
    dict_comp_one = {k: 1 for k in keys} # Noncompliant
    dict_comp_one = {k: "1" for k in keys} # Noncompliant
    dict_comp_one = {k: True for k in keys} # Noncompliant
    dict_comp_none = {k: None for k in keys} # Noncompliant
    dict_comp_v = {k: some_v for k in keys} # Noncompliant

    some_dict = {"a": "1", "b": "2"}
    one_more = {k: some_v for k, v in some_dict} # Noncompliant
    one_more = {k: v for k, v in some_dict}

def case2():
    dict_of_dicts = [{"a": 1, "b": 2}, {"c": 3, "d": 4}]
    result = {k: v for result in dict_of_dicts for k, v in result.items()}

def case3():
    some_list = ["a/b", "c/d"]
    result = {q.split("/")[-1]: q for q in some_list}

def case4(columns_to_save):
    mapping = {col: f"new.{col}" for col in columns_to_save}
    some_v = "v"
    some_dict = {"a": "1", "b": "2"}
    one_more = {k: f"{some_v}" for k, v in some_dict} # FN
