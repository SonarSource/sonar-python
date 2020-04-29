# Mutably borrowed from
# security-expected-issues/python/rules/hotspots/RSPEC-4502/flask/globalsensitive1.py
from flask import Flask
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
csrf = CSRFProtect()
app.config['WTF_CSRF_ENABLED'] = True # For code coverage only, shouldn't report anything here.
app.config['WTF_CSRF_ENABLED'] = False # Noncompliant {{Disabling CSRF protection is dangerous.}}
csrf.init_app(app) # for example here
