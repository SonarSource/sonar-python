def foo(param): # Noncompliant {{Add a type hint to this function parameter.}}
       #^^^^^
    pass

def foobar_multiline(
        param1, # Noncompliant {{Add a type hint to this function parameter.}}
       #^^^^^^
        param2): # Noncompliant {{Add a type hint to this function parameter.}}
       #^^^^^^
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

def nested(param1: str):
    def foo(param2): # Noncompliant {{Add a type hint to this function parameter.}}
           #^^^^^^
        pass
    print("the end")

def dynamic_param_list(*args, **kwargs): # Noncompliant 2
    pass

def self_outside_cls(self): # Noncompliant {{Add a type hint to this function parameter.}}
                    #^^^^
    pass

class SuccessBar:
    def __init__(self, param1: str):
        pass

    @classmethod
    def class_method(cls):
        pass

    def not_class_method(cls: str):
        pass

    def no_params():
        pass

    def foo(param1: str):
        pass

    def foobar(param1: str, param2: int):
        pass

def union(param1: str|int):
    pass

def success_dynamic_param_list(*args: str, **kwargs: int):
    pass
