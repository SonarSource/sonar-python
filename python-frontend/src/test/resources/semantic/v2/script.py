import datetime
import lib

# builtins
my_string_literal = "hello"
my_number_literal = 42
my_list = ["a", "b", "c"]
my_list.append("d")

# on prem. class declaration
class MyClass:
    def my_method(self):
        return 42

# on prem class object instantiation
my_object = MyClass()
b = my_object.my_method()

# on prem function declaration
def script_do_something(param):
    return 42

# function call
c = script_do_something(42)


# project module type information
d = lib.lib_function()
lib_obj = lib.LibClass()
e = lib_obj.do_something()

# lib module type information
f = datetime.date()
f.replace(year = 2023)

# nested type information
g = datetime.date.today()

# from import
from lib import lib_function
lib_function()