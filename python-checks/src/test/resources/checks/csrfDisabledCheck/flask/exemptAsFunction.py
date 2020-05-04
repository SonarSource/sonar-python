def exemptExample():
  from flask import Flask, Blueprint
  from flask_wtf.csrf import CSRFProtect
  from simplepages import simple_page
   
  app = Flask(__name__)
  app.register_blueprint(simple_page)
  
  csrf = CSRFProtect()
  csrf.init_app(app) 
  csrf.exempt(simple_page) # Noncompliant {{Make sure disabling CSRF protection is safe here.}}
  #    ^^^^^^
  
