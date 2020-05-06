def csrfInitAppShouldWorkEvenIfCsrfSymbolIsNotFound():
  from flask import Flask
  app = Flask(__name__) # Compliant
  csrf.init_app(app) # Unknown symbol, but looks similar enough to CSRFProtect.

def csrfInitAppShouldWorkEvenIfCsrfSymbolIsNotFoundUppercase():
  from flask import Flask
  app = Flask(__name__) # Compliant
  CSRF.init_app(app) # Unknown symbol, but looks similar enough to CSRFProtect.

def csrfInitAppShouldCheckTheQualifier():
  from flask import Flask
  app = Flask(__name__) # Noncompliant {{Make sure disabling CSRF protection is safe here.}}
  #     ^^^^^^^^^^^^^^^
  somethingUnrelated.init_app(app) # insufficient
  tooLong.csrf.init_app(app) # insufficient
  csrf.do_something_else(app) # insufficient

def csrfProtectCanBeImportedFromFlaskWtfDirectly():
  from flask import Flask
  from flask_wtf import CSRFProtect
  app = Flask(__name__) # Compliant
  CSRFProtect(app)

