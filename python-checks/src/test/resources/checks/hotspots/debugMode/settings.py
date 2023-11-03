DEBUG = True  # Noncompliant
DEBUG_PROPAGATE_EXCEPTIONS = True  # Noncompliant
DEBUG += True  # OK
DEBUG = False  # OK
DEBUG = print()  # OK
DEBUG_PROPAGATE_EXCEPTIONS = False  # OK
Other = True

def my_method(hello):
    hello.errors = True

def flask_test():
    from flask import Flask
    app = Flask()

    app.debug = True  # Noncompliant
