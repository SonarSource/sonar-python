
def case1():
  some_dict = { "hi": "hello"}
  for k, v in some_dict: # Noncompliant {{Use items to iterate over key-value pairs}}
    #         ^^^^^^^^^
    ...
  {k: v for k, v in some_dict} # Noncompliant {{Use items to iterate over key-value pairs}}
  #                 ^^^^^^^^^

def case2():
  some_dict = { "hi": "hello"}
  for k, v in some_dict.items():
    ...
  {k: v for k, v in some_dict.items()}

def case3():
  not_a_dict = ["hi", "hello"]
  for k, v in not_a_dict:
    ...
  {k: v for k, v in not_a_dict}

def case4():
  some_dict = { "hi": "hello"}
  for k in some_dict:
    ...
  for k, v in some_dict, something_else:
    ...
  {k: 1 for k in some_dict}
  {k: 1 for k, v, i in not_a_dict}