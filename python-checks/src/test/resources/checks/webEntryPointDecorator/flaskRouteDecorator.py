from flask import Flask
from flask import Blueprint

app = Flask(__name__)
bp = Blueprint('bp', __name__)

def login_required(f):
    return f

def cache_timeout(timeout):
    def decorator(f):
        return f
    return decorator

@login_required
@app.route('/secret_page')  # Noncompliant {{Move this '@route' decorator to the top of the other decorators.}}
def secret_page():
    return "Secret content"

@app.route('/public_page')
@login_required
def public_page():
    return "Public content"

@cache_timeout(300)
@login_required
@app.route('/dashboard')  # Noncompliant
def dashboard():
    return "Dashboard"

@login_required
@app.route('/middle')  # Noncompliant
@cache_timeout(300)
def route_in_middle():
    return "Route in middle"

@app.route('/cached')
@login_required
@cache_timeout(300)
def cached_view():
    return "Cached"

@app.route('/simple')
def simple_view():
    return "Simple"

@login_required
@bp.route('/bp_secret')  # Noncompliant
def bp_secret_page():
    return "Blueprint secret"

@bp.route('/bp_public')
@login_required
def bp_public_page():
    return "Blueprint public"

@login_required
@cache_timeout(300)
def helper_function():
    return "Helper"

