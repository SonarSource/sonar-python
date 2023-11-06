from flask import Flask, current_app
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



def test_non_compliant_assignment_expressions():
    app = Flask(__name__)

    assigned_secret = 'hardcoded_secret'
    # Tests for "flask.app.Flask.config"
    app.config['SECRET_KEY'] = 'secret'  # Noncompliant

    # Tests for "flask.app.Flask.secret_key"
    app.secret_key = 'mysecret'  # Noncompliant

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

def test_compliant(x, args):

    current_app.config.update(dict(
        SECRET_KEY=x  # OK, but would like to double check about this.
    ))

    assigned_secret = 'hardcoded_secret'

    current_app.config['SECRET'] = 'secret'  # OK
    current_app.config['SECRET'] = assigned_secret  # OK
    x = current_app.config['SECRET'] = 'secret'  # OK
    current_app.config['SECRET'] = x = 'secret'  # OK
    current_app.config['SECRET'], _ = 'mysecret', x # OK
    current_app.config['SECRET'], _ = _, current_app.config['SECRET'] = 'secret', x # OK

    current_app.config['SECRET'] = x  # OK
    y, current_app.config['SECRET'] = 'some_random_string', x
    current_app.config['SECRET'], y = z, current_app.config['SECRET'] = x, 'secret'

    current_app.config['SECRET'], y = z, current_app.config['SECRET'] = x

