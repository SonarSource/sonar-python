unread_global = 1

def f(unread_param):
    global unread_global
    unread_global = 1
    unread_local = 1 # Noncompliant {{Remove the unused local variable "unread_local".}}
    unread_local = 2 # Noncompliant
    read_local = 1
    print(read_local)
    read_in_nested_function = 1
    def nested_function():
        print(read_in_nested_function)

def using_locals(a, b):
  c = a + b
  # "locals" will include the "c" value
  return locals()

def string_interpolation():
    value1 = 1
    value2 = 2 # Noncompliant
    value3 = 3 # false-negative, value3 is not used as a variable in the string interpolation, see SONARPY-245
    return f'The value1 is {value1}, value2bis, value3'
