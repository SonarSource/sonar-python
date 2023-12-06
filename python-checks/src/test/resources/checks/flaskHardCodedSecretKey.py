from flask import Flask, current_app
def test_non_compliant_assignment_expressions(x):
    app = Flask(__name__)
    assigned_secret = 'hardcoded_secret'

    # Tests for "flask.app.Flask.config"
    app.config['SECRET_KEY'] = 'secret'  # Noncompliant {{Don't disclose "Flask" secret keys.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^>                           {{Assignment to sensitive property.}}
#                              ^^^^^^^^@-1
    app.config['SECRET_KEY'] = assigned_secret  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^>  ^^^^^^^^^^^^^^^
    _ = app.config['SECRET_KEY'] = 'secret'  # Noncompliant
#       ^^^^^^^^^^^^^^^^^^^^^^^^>  ^^^^^^^^

    #   False negatives.
    app.config['SECRET_KEY'], _ = 'secret', x # FN: Should be extended to ExpressionList in the lhs containing more than one expression
    app.config['SECRET_KEY'], _ = _, app.config['SECRET_KEY'] = 'secret', x # FN: Same as above


    # Tests for "flask.app.Flask.secret_key"
    app.secret_key = 'secret'         # Noncompliant {{Don't disclose "Flask" secret keys.}}
#   ^^^^^^^^^^^^^^>                                    {{Assignment to sensitive property.}}
#                    ^^^^^^^^@-1
    app.secret_key = assigned_secret  # Noncompliant
    _ = app.secret_key = 'mysecret'  # Noncompliant

    #   False negatives.
    app.secret_key, _ = 'secret', x # FN: Should be extended to ExpressionList in the lhs containing more than one expression
    app.secret_key, _ = _, app.secret_key = 'secret', x # FN: Same as above


    # Tests for "flask.globals.current_app.config"
    current_app.config['SECRET_KEY'] = 'secret'  # Noncompliant {{Don't disclose "Flask" secret keys.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>                                    {{Assignment to sensitive property.}}
#                                      ^^^^^^^^@-1
    current_app.config['SECRET_KEY'] = assigned_secret  # Noncompliant
    _ = current_app.config['SECRET_KEY'] = 'secret'  # Noncompliant

    #    False negatives.
    current_app.config['SECRET_KEY'], _ = 'mysecret', x # FN
    current_app.config['SECRET_KEY'], _ = _, current_app.config['SECRET_KEY'] = 'secret', x # FN


    # Tests for "flask.globals.current_app.secret_key"
    current_app.secret_key = 'secret'   # Noncompliant {{Don't disclose "Flask" secret keys.}}
#   ^^^^^^^^^^^^^^^^^^^^^^>                            {{Assignment to sensitive property.}}
#                            ^^^^^^^^@-1
    current_app.secret_key = assigned_secret  # Noncompliant
    _ = current_app.secret_key = 'mysecret'  # Noncompliant

    #   False negatives.
    current_app.secret_key, _ = assigned_secret, x # FN
    current_app.secret_key, _ = _, current_app.secret_key = 'secret', x # FN

    current_app.config['SECRET_KEY'] = app.config['SECRET_KEY'] = 'secret'      # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>  ^^^^^^^^^^^^^^^^^^^^^^^^>  ^^^^^^^^
    some_secret = assigned_secret
    some_secret1 = some_secret
    some_secret2 = some_secret1
    some_secret3 = some_secret2
    some_secret4 = some_secret3
    some_secret5 = some_secret4
    y = some_secret5
    app.secret_key = y  # Noncompliant




def test_non_compliant_call_expressions():
    app = Flask(__name__)

    # Tests for "flask.app.Flask.config[.update]"
    app.config.update(dict(
        SECRET_KEY="woopie" # Noncompliant
    ))

    app.config.update({"SECRET_KEY": "woopie"}) # Noncompliant

    d = dict(
        SECRET_KEY="woopie" # Noncompliant 2
    )
    d1 = {"SECRET_KEY": "woopie"} # Noncompliant 2

    app.config.update(d)
    app.config.update(d1)

    # Tests for "flask.globals.current_app.config.update"
    current_app.config.update(dict(
        SECRET_KEY="woopie" # Noncompliant
    ))

    current_app.config.update({"SECRET_KEY": "woopie"}) # Noncompliant

    current_app.config.update(d)
    current_app.config.update(d1)

    d2 = dict(SECRET_KEY="woopie") # Noncompliant {{Don't disclose "Flask" secret keys.}}
#             ^^^^^^^^^^^^^^^^^^^
    app.config.update(d2)
#   ^^^^^^^^^^^^^^^^^<1 {{The secret is used in this call.}}
    d3 = {"SECRET_KEY": "woopie"} # Noncompliant {{Don't disclose "Flask" secret keys.}}
#         ^^^^^^^^^^^^^^^^^^^^^^
    app.config.update(d3)
#   ^^^^^^^^^^^^^^^^^<1 {{The secret is used in this call.}}


def get_secret_from_vault():
    pass


def test_compliant(x):
    app = Flask(__name__)
    assigned_secret = 'hardcoded_secret'

    app.config['A_KEY'] = 'secret'  # OK
    app.config['A_KEY'] = assigned_secret  # OK
    _ = app.config['A_KEY'] = 'secret'  # OK
    app.config['A_KEY'], _ = 'mysecret', x # OK
    app.config['A_KEY'], _ = _, app.config['SECRET_KEY'] = 'secret', x # OK

    d = dict(
        A_KEY="woopie"
    )
    d1 = {"A_KEY": "woopie"}
    d2 = {"SECRET_KEY": x}

    app.config.update(d)
    app.config.update(d1)
    app.config.update(d2)

    current_app.config.update(dict(
        SECRET_KEY=x  # OK
    ))
    def hello(SECRET_KEY):
        ...

    current_app.config.update(dict(
        SECRET_KEY=hello('something')  # OK
    ))

    current_app.config.update(dict(
        A_KEY='something'  # OK
    ))

    current_app.config['SECRET_KEY'] = x

    current_app.config.update(hello(
        SECRET_KEY='SECRET'  # OK
    ))
    current_app.config.update()
    current_app.config.update('secret')

    current_app.config[x] = 'secret'  # OK
    current_app.config['A_KEY'] = 'secret'  # OK
    current_app.config['A_KEY', x] = 'secret'  # OK
    x = current_app.config['A_KEY'] = 'secret'  # OK
    current_app.config['A_KEY'], _ = 'mysecret', x # OK
    current_app.config['A_KEY'], _ = _, current_app.config['SECRET_KEY'] = 'secret', x # OK

    current_app.config['A_KEY'] = x  # OK
    y, current_app.config['A_KEY'] = 'some_random_string', x
    current_app.config['A_KEY'], y = z, current_app.config['A_KEY'] = x, 'secret'

    current_app.config['A_KEY'], y = z, current_app.config['A_KEY'] = x

    secret = 'secret'

    secret = get_secret_from_vault()

    current_app.secret_key = secret # Compliant

    some_secret = x
    some_secret1 = some_secret
    some_secret2 = some_secret1
    some_secret3 = some_secret2
    some_secret4 = some_secret3
    some_secret5 = some_secret4
    x = some_secret5
    app.secret_key = x # OK: x doesn't have a single assigned value.

x = []
class TestInfiniteRecursion():
    x = x
    current_app.secret_key = x  # Coverage: no more infinite recursion.
