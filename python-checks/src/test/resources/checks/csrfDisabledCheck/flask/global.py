def misconfiguredFlaskExamples():
  from flask import Flask
  from flask_wtf.csrf import CSRFProtect
  # from flask_wtf import csrf

  app1 = Flask(__name__) # Noncompliant {{CSRFProtect is missing.}}
  #      ^^^^^^^^^^^^^^^

  app2 = Flask(__name__)
  c2 = CSRFProtect()
  c2.init_app(app2) # Compliant

  app3 = Flask(__name__)
  app3.config['SECRET_KEY'] = 'top-secret!'
  c3 = CSRFProtect(app3) # Compliant

  @app3.route('/') # it's here just to make sure we don't somehow crash on usages within decorators
  def index():
      pass
