print("aa a") # ok, too short
print("aa a")
print("aa a")

print("\x61\x61\x20\x61") # ok, too short
print("\x61\x61\x20\x61")
print("\x61\x61\x20\x61")

print("aa aa") # Noncompliant {{Define a constant instead of duplicating this literal "aa aa" 3 times.}} [[effortToFix=2]]
#     ^^^^^^^
print("aa aa")
#     ^^^^^^^<
print("aa aa")
#     ^^^^^^^<

print("bb bb") # ok, only 2 literals
print("bb bb")

print(r"bb\\bb") # Noncompliant
print(r"bb\\bb")
print(r"bb\\bb")

print("aa\\aa") # compliant
print("aa\\aa")
print(r"aa\\aa")

def literals_with_multiple_elements():
    print("First element of a string literal." # Noncompliant 
          "Second element of a string literal.")
#         ^[ec=47;el=+1]@-1
    print("First element of a string literal."
          "Second element of a string literal.")
    print("First element of a string literal."
          "Second element of a string literal.")
    print("First element"
          "Second element 1")
    print("First element"
          "Second element 1")
    print("First element"
          "Second element 2")

def fstrings_should_be_excluded():
    name = "Alice"
    print(f"hello {name}")
    print(f"hello {name}")
    name = "Bob"
    print(f"hello {name}")

def decorators_should_be_excluded():
    from flask import Flask
    app = Flask(__name__)

    @app.route("/api/users/", methods=['GET'])
    def users_get():
        pass

    @app.route("/api/users/", methods=['PUT'])
    def users_put():
        pass

    @app.route("/api/users/", methods=['POST'])
    def users_post():
        pass

def literals_with_only_letters_and_digits_and_underscore_or_dash_should_be_excluded():
    options = { 'debug_logs': False }
    if options['debug_logs']:
        print(options['debug_logs'])
    encoding = 'utf-8'
    encoding = 'utf-8'
    encoding = 'utf-8'

def color_patterns_should_be_excluded():
  color = "#ffffcc"
  color2 = "#ffffcc"
  color3 = "#ffffcc"

  not_a_color = "#ffffcca" #Noncompliant
  not_a_color = "#ffffcca"
  not_a_color = "#ffffcca"

def simple_formatting_patterns_should_be_excluded(args):
    print("%s %s" % args)
    print("%s %s" % args)
    print("%s %s" % args)
    print('{:>10}'.format(args))
    print('{:>10}'.format(args))
    print('{:>10}'.format(args))

def docstrings_should_be_excluded():
    def duplicated_docstring_1():
        """This is a duplicated docstring"""
        pass

    def duplicated_docstring_2():
        """This is a duplicated docstring"""
        pass

    def duplicated_docstring_3():
        """This is a duplicated docstring"""
        pass

def type_hints_should_be_excluded():
    a: "httpx.Response"
    b: "httpx.Response"
    c: "httpx.Response"

def literal_patterns_should_be_excluded(value1, value2, value3):
    match value1:
        case "This is a duplicated pattern":
            print("This is a duplicated literal")  # Noncompliant

    match value2:
        case "This is a duplicated pattern":
            print("This is a duplicated literal")

    match value3:
        case "This is a duplicated pattern":
            print("This is a duplicated literal")

print("THIS IS A STRING LITERAL") # Noncompliant
print("THIS IS A STRING LITERAL")
print("THIS IS A STRING LITERAL")
