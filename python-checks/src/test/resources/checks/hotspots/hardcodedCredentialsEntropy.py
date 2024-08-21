def func():
    token = "rf6acB24J//1FZLRrKpjmBUYSnUX5CHlt/iD5vVVcgVuAIOB6hzcWjDnv16V6hDLevW0Qs4hKPbP1M4YfuDI16sZna1/VGRLkAbTk6xMPs4epH6A3ZqSyyI-H92y" # Noncompliant
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    api_key = "not enough entropy"
    api_key = "rf6acB24J//1FZLRrKpjmBUYSnUX5CHlt/iD5vVVcgVuAIOB6hzcWjDnv16V6hDLevW0Qs4hKPbP1M4YfuDI16sZna1/VGRLkAbTk6xMPs4epH6A3ZqSyyI-H92y" # Noncompliant

def entropy_too_low():
    token = "rf6acB24J//1FZLRrKpjmBUYSnUX5CHlt/iD5vVaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

class MyClass:
    secret = "1IfHMPanImzX8ZxC-Ud6+YhXiLwlXq$f_-3v~.=" # Noncompliant {{"secret" detected here, make sure this is not a hard-coded secret.}}


def in_function_call():
    call_with_secret(secret="1IfHMPanImzX8ZxC-Ud6+YhXiLwlXq$f_-3v~.=") # Noncompliant

def function_with_secret(secret="1IfHMPanImzX8ZxC-Ud6+YhXiLwlXq$f_-3v~.="): # Noncompliant
#                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    pass

def clean_function(some_arg, parameter="a string", another_parameter: 42, *args, **kwargs):
    another_call(42, "a string", parameter, *args)


some_dict = {
    "secret": "1IfHMPanImzX8ZxC-Ud6+YhXiLwlXq$f_-3v~.=", # Noncompliant
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    "not_a_problem": "not_a_secret",
    42: "forty-two"
}

def multiple_assignment():
    nothing, secret, nothing_else = 1, "1IfHMPanImzX8ZxC-Ud6+YhXiLwlXq$f_-3v.~=", 2 # Noncompliant
                                      #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def assignment_with_type():
    secret: str = "1IfHMPanImzX8ZxC-Ud6+YhXiLwlXq$f_-3v~.=" # Noncompliant

    some_var: str
    another_var: int = 42
