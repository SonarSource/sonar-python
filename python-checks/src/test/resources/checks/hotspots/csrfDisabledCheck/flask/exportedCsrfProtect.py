from flask import Flask
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
csrfProtect = CSRFProtect()
csrfProtect.init_app(app) # Compliant
