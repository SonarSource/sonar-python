import another_module

data = ["some", "Data", "example"]
my_dict = {"key": value.upper() for value in data} # Noncompliant {{Don't use a static key in a dictionary comprehension.}}
#          ^^^^^

key_literal = "key"
my_dict = {key_literal: value.upper() for value in data} # Noncompliant {{Don't use a static key in a dictionary comprehension.}}
#          ^^^^^^^^^^^

another_key_literal = key_literal
my_dict = {another_key_literal: value.upper() for value in data} # Noncompliant {{Don't use a static key in a dictionary comprehension.}}
#          ^^^^^^^^^^^^^^^^^^^

f_string_key = f"{value}_str"
another_f_string_key = f_string_key
my_dict = {another_f_string_key: value.upper() for value in data} # Compliant

my_dict = {value: value.upper() for value in data} # Compliant

key_from_another_module = another_module.key
my_dict = {key_from_another_module: value.upper() for value in data} # Compliant

my_dict = {f"{value}_str": value.upper() for value in data} # Compliant

my_dict = {f"{key_literal}_str": value.upper() for value in data} # FN : interpolated string is constant
