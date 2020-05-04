# Mutably borrowed from
# security-expected-issues/python/rules/hotspots/RSPEC-4502/flask/globalsensitive1.py
def wtfCsrfEnabledExamples():
  from flask import Flask
  from flask_wtf.csrf import CSRFProtect

  app = Flask(__name__)
  app.config['WTF_CSRF_ENABLED'] = True # For code coverage only, shouldn't report anything here.
  app.config['WTF_CSRF_ENABLED'] = False # Noncompliant {{Make sure disabling CSRF protection is safe here.}}
  #                                ^^^^^
  csrfProtect = CSRFProtect()
  csrfProtect.init_app(app)

  app2 = Flask(__name__)
  app2.config['WTF_CSRF_CHECK_DEFAULT'] = False # Noncompliant {{Make sure disabling CSRF protection is safe here.}}
  #                                       ^^^^^

  CSRFProtect(app2)
