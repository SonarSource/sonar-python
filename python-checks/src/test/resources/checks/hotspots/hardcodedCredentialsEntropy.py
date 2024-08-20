def func():
    token = "rf6acB24J//1FZLRrKpjmBUYSnUX5CHlt/iD5vVVcgVuAIOB6hzcWjDnv16V6hDLevW0Qs4hKPbP1M4YfuDI16sZna1/VGRLkAbTk6xMPs4epH6A3ZqSyyI-H92y" # Noncompliant
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    api_key = "not enough entropy"
    api_key = "rf6acB24J//1FZLRrKpjmBUYSnUX5CHlt/iD5vVVcgVuAIOB6hzcWjDnv16V6hDLevW0Qs4hKPbP1M4YfuDI16sZna1/VGRLkAbTk6xMPs4epH6A3ZqSyyI-H92y" # Noncompliant

class MyClass:
    secret = "1IfHMPanImzX8ZxC-Ud6+YhXiLwlXq$f_-3v~.=" # Noncompliant {{"secret" detected here, make sure this is not a hard-coded secret.}}


def in_function_call():
    call_with_secret(secret="1IfHMPanImzX8ZxC-Ud6+YhXiLwlXq$f_-3v~.=") # Noncompliant

def function_with_secret(secret="1IfHMPanImzX8ZxC-Ud6+YhXiLwlXq$f_-3v~.="): # Noncompliant
#                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    pass

some_dict = {
    "secret": "1IfHMPanImzX8ZxC-Ud6+YhXiLwlXq$f_-3v~.=" # Noncompliant
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
}

def multiple_assignment():
    nothing, secret, nothing_else = 1, "1IfHMPanImzX8ZxC-Ud6+YhXiLwlXq$f_-3v.~=", 2 # Noncompliant
                                      #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^