from flask import Flask, current_app
def test_non_compliant_assignment_expressions(x):
    app = Flask(__name__)
    assigned_secret = 'hardcoded_secret'

    # Tests for "flask.app.Flask.config"
    app.config['JWT_SECRET_KEY'] = 'secret'  # Noncompliant {{Don't disclose "Flask" JWT secret keys.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^>                           {{Assignment to sensitive property.}}
#                                  ^^^^^^^^@-1
    app.config['JWT_SECRET_KEY'] = assigned_secret  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^>  ^^^^^^^^^^^^^^^
    _ = app.config['JWT_SECRET_KEY'] = 'secret'  # Noncompliant
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^>  ^^^^^^^^

    #   False negatives.
    app.config['JWT_SECRET_KEY'], _ = 'secret', x # FN: Should be extended to ExpressionList in the lhs containing more than one expression
    app.config['JWT_SECRET_KEY'], _ = _, app.config['JWT_SECRET_KEY'] = 'secret', x # FN: Same as above


    # Tests for "flask.globals.current_app.config"
    current_app.config['JWT_SECRET_KEY'] = 'secret'  # Noncompliant {{Don't disclose "Flask" JWT secret keys.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>                           {{Assignment to sensitive property.}}
#                                          ^^^^^^^^@-1
    current_app.config['JWT_SECRET_KEY'] = assigned_secret  # Noncompliant
    _ = current_app.config['JWT_SECRET_KEY'] = 'secret'  # Noncompliant

    #   False negatives.
    current_app.config['JWT_SECRET_KEY'], _ = 'mysecret', x # FN
    current_app.config['JWT_SECRET_KEY'], _ = _, current_app.config['JWT_SECRET_KEY'] = 'secret', x # FN


    current_app.config['JWT_SECRET_KEY'] = app.config['JWT_SECRET_KEY'] = 'secret'      # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^>  ^^^^^^^^



def test_non_compliant_call_expressions():
    app = Flask(__name__)

    # Tests for "flask.app.Flask.config[.update]"
    app.config.update(dict( # Noncompliant
#   ^[el=+3;ec=6]
        JWT_SECRET_KEY="woopie"
    ))

    app.config.update({"JWT_SECRET_KEY": "woopie"}) # Noncompliant

    d = dict(
        JWT_SECRET_KEY="woopie"
    )
    d1 = {"JWT_SECRET_KEY": "woopie"}

    app.config.update(d)            # Noncompliant
    app.config.update(d1)           # Noncompliant

    # Tests for "flask.globals.current_app.config.update"
    current_app.config.update(dict( # Noncompliant
        JWT_SECRET_KEY="woopie"
    ))

    current_app.config.update({"JWT_SECRET_KEY": "woopie"}) # Noncompliant

    current_app.config.update(d)            # Noncompliant
    current_app.config.update(d1)           # Noncompliant


def get_secret_from_vault():
    pass


def test_compliant(x):
    app = Flask(__name__)
    assigned_secret = 'hardcoded_secret'

    app.config['A_KEY'] = 'secret'  # OK
    app.config['A_KEY'] = assigned_secret  # OK
    x = app.config['A_KEY'] = 'secret'  # OK
    app.config['A_KEY'], _ = 'mysecret', x # OK
    app.config['A_KEY'], _ = _, app.config['JWT_SECRET_KEY'] = 'secret', x # OK

    d = dict(
        A_KEY="woopie"
    )
    d1 = {"A_KEY": "woopie"}

    app.config.update(d)
    app.config.update(d1)

    current_app.config.update(dict(
        JWT_SECRET_KEY=x  # OK
    ))
    def hello(JWT_SECRET_KEY):
        ...

    current_app.config.update(dict(
        JWT_SECRET_KEY=hello('something')  # OK
    ))

    current_app.config.update(dict(
        A_KEY='something'  # OK
    ))

    current_app.config['JWT_SECRET_KEY'] = x

    current_app.config.update(hello(
        JWT_SECRET_KEY='SECRET'  # OK
    ))
    current_app.config.update()
    current_app.config.update('secret')

    current_app.config[x] = 'secret'  # OK
    current_app.config['A_KEY'] = 'secret'  # OK
    current_app.config['A_KEY', x] = 'secret'  # OK
    x = current_app.config['A_KEY'] = 'secret'  # OK
    current_app.config['A_KEY'], _ = 'mysecret', x # OK
    current_app.config['A_KEY'], _ = _, current_app.config['JWT_SECRET_KEY'] = 'secret', x # OK

    current_app.config['A_KEY'] = x  # OK
    y, current_app.config['A_KEY'] = 'some_random_string', x
    current_app.config['A_KEY'], y = z, current_app.config['A_KEY'] = x, 'secret'

    current_app.config['A_KEY'], y = z, current_app.config['A_KEY'] = x

    secret = 'secret'

    secret = get_secret_from_vault()

    current_app.secret_key = secret # Compliant

