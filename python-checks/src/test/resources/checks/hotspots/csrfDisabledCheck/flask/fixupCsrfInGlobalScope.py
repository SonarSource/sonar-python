from flask import Flask
from flask_wtf import CSRFProtect
csrf = CSRFProtect()
def create_app():
   app = Flask(__name__) # Compliant
   csrf.init_app(app)

def create_app_noncompliant():
   app2 = Flask(__name__) # Noncompliant

