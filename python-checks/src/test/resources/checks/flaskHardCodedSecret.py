from flask import Flask, current_app
def test_non_compliant_assignment_expressions(x):
    app = Flask(__name__)
    assigned_secret = 'hardcoded_secret'

    # Tests for "flask.app.Flask.config"
    app.config['SECRET_KEY'] = 'secret'  # Noncompliant
    app.config['SECRET_KEY'] = assigned_secret  # Noncompliant
    x = app.config['SECRET_KEY'] = 'secret'  # Noncompliant
    app.config['SECRET_KEY'] = x = 'secret'  # Noncompliant
    app.config['SECRET_KEY'], _ = 'mysecret', x # Noncompliant
    app.config['SECRET_KEY'], _ = _, app.config['SECRET_KEY'] = 'secret', x # Noncompliant


    # Tests for "flask.app.Flask.secret_key"
    app.secret_key = 'mysecret'  # Noncompliant
    app.secret_key = assigned_secret  # Noncompliant
    _ = app.secret_key = 'mysecret'  # Noncompliant
    app.secret_key = _ = 'mysecret'  # Noncompliant

    # Tests for "flask.globals.current_app.config"
    current_app.config['SECRET_KEY'] = 'secret'  # Noncompliant
    current_app.config['SECRET_KEY'] = assigned_secret  # Noncompliant
    x = current_app.config['SECRET_KEY'] = 'secret'  # Noncompliant
    current_app.config['SECRET_KEY'] = x = 'secret'  # Noncompliant
    current_app.config['SECRET_KEY'], _ = 'mysecret', x # Noncompliant
    current_app.config['SECRET_KEY'], _ = _, current_app.config['SECRET_KEY'] = 'secret', x # Noncompliant

    # Tests for "flask.globals.current_app.secret_key"
    current_app.secret_key = 'mysecret'  # Noncompliant
    current_app.secret_key = assigned_secret  # Noncompliant
    _ = current_app.secret_key = 'mysecret'  # Noncompliant
    current_app.secret_key = _ = 'mysecret'  # Noncompliant

    current_app.secret_key, _ = assigned_secret, x # Noncompliant
    current_app.secret_key, _ = _, current_app.secret_key = 'secret', x # Noncompliant
def test_non_compliant_call_expressions():
    app = Flask(__name__)

    # Tests for "flask.app.Flask.config[.update]"
    app.config.update(dict( # Noncompliant
        SECRET_KEY="woopie"
    ))

    app.config.update({"SECRET_KEY": "woopie"}) # Noncompliant

    d = dict(
        SECRET_KEY="woopie"
    )
    d1 = {"SECRET_KEY": "woopie"}

    app.config.update(d)            # Noncompliant
    app.config.update(d1)           # Noncompliant

    # Tests for "flask.globals.current_app.config.update"
    current_app.config.update(dict( # Noncompliant
        SECRET_KEY="woopie"
    ))

    current_app.config.update({"SECRET_KEY": "woopie"}) # Noncompliant

    current_app.config.update(d)            # Noncompliant
    current_app.config.update(d1)           # Noncompliant


def test_compliant(x):
    app = Flask(__name__)
    assigned_secret = 'hardcoded_secret'

    app.config['SECRET'] = 'secret'  # OK
    app.config['SECRET'] = assigned_secret  # OK
    x = app.config['SECRET'] = 'secret'  # OK
    app.config['SECRET'] = x = 'secret'  # OK
    app.config['SECRET'], _ = 'mysecret', x # OK
    app.config['SECRET'], _ = _, app.config['SECRET_KEY'] = 'secret', x # OK

    current_app.config.update(dict(
        SECRET_KEY=x  # OK, but not sure if it should be.
    ))
    def hello(SECRET_KEY):
        ...

    current_app.config.update(dict(
        SECRET_KEY=hello('something')  # OK, but not sure if it should be.
    ))


    current_app.config.update(hello(
        SECRET_KEY='SECRET'  # OK, but not sure if it should be.
    ))
    current_app.config.update()
    current_app.config.update('secret')

    assigned_secret = 'hardcoded_secret'

    current_app.config[x] = 'secret'  # OK
    current_app.config['SECRET'] = 'secret'  # OK
    current_app.config['SECRET', x] = 'secret'  # OK
    current_app.config['SECRET'] = assigned_secret  # OK
    x = current_app.config['SECRET'] = 'secret'  # OK
    current_app.config['SECRET'] = x = 'secret'  # OK
    current_app.config['SECRET'], _ = 'mysecret', x # OK
    current_app.config['SECRET'], _ = _, current_app.config['SECRET'] = 'secret', x # OK

    current_app.config['SECRET'] = x  # OK
    y, current_app.config['SECRET'] = 'some_random_string', x
    current_app.config['SECRET'], y = z, current_app.config['SECRET'] = x, 'secret'

    current_app.config['SECRET'], y = z, current_app.config['SECRET'] = x

