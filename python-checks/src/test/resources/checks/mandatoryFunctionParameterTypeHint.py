def foo(param): # Noncompliant {{Add a type hint to this function parameter.}}
       #^^^^^
    pass

def foobar_multiline(
        param1, # Noncompliant {{Add a type hint to this function parameter.}}
       #^^^^^^
        param2): # Noncompliant {{Add a type hint to this function parameter.}}
       #^^^^^^
    pass

def format(number, decimal_sep, decimal_pos=None, grouping=0, thousand_sep='', # Noncompliant 5
           force_grouping=False, use_l10n=None): # Noncompliant 2
    pass

def set(self): # Noncompliant {{Add a type hint to this function parameter.}}
       #^^^^
    pass

def exit(cls): # Noncompliant {{Add a type hint to this function parameter.}}
        #^^^
    pass

def foobar(param1, param2): # Noncompliant 2
    pass


class Bar:
    def __init__(self, param): # Noncompliant {{Add a type hint to this function parameter.}}
                      #^^^^^
        pass

    def foo(param): # Noncompliant {{Add a type hint to this function parameter.}}
           #^^^^^
        pass

    def not_class_method(cls): # Noncompliant {{Add a type hint to this function parameter.}}
                        #^^^
        pass

    @classmethod
    def class_method(cls, param): # Noncompliant {{Add a type hint to this function parameter.}}
                         #^^^^^
        pass

    def __new__(self, param2): # Noncompliant {{Add a type hint to this function parameter.}}
                     #^^^^^^
        pass

def nested(param1: str):
    def foo(param2): # Noncompliant {{Add a type hint to this function parameter.}}
           #^^^^^^
        pass
    print("the end")

def dynamic_param_list(param1, param2, *args, **kwargs): # Noncompliant 4
    pass

def dynamic_args(param:str, *args): # Noncompliant {{Add a type hint to this function parameter.}}
                           #^^^^^
    pass

def dynamic_kwargs(param:int, **kwargs): # Noncompliant {{Add a type hint to this function parameter.}}
                             #^^^^^^^^
    pass

def default_params(param=None): # Noncompliant {{Add a type hint to this function parameter.}}
                  #^^^^^^^^^^
    pass

def self_outside_cls(self): # Noncompliant {{Add a type hint to this function parameter.}}
                    #^^^^
    pass

def __init__(self, param1: str): # Noncompliant {{Add a type hint to this function parameter.}}
            #^^^^
    pass


class SuccessBar:
    def __init__(self, param1: str):
        pass

    @classmethod
    def class_method(cls):
        pass

    @classmethod
    def class_method_first_param(class_param, param1: str, param2:int):
        pass

    def __new__(cls, param2:str):
        pass

    def not_class_method(cls: str):
        pass

    def no_params():
        pass

    def foo(param1: str):
        pass

    def foobar(param1: str, param2: int):
        pass

    def __enter__(self):
        pass

    def __exit__(self):
        pass

def union(param1: str|int):
    pass

def success_dynamic_param_list(*args: str, **kwargs: int):
    pass

def success_slash(param1: str, /, param2:int):
    pass

def success_star(param1: str, *, param2:int):
    pass

def success_underscore(_):
    pass

