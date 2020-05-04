from flask import Flask
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
app.config['SECRET_KEY'] = 'top-secret!'
csrf = CSRFProtect()
csrf.init_app(app) # Compliant

@app.route('/csrftest1/', methods=['POST'])
@csrf.exempt # Noncompliant {{Make sure disabling CSRF protection is safe here.}}
#     ^^^^^^
def csrftestpost():
    pass

# Corner cases for test coverage
@csrf.thisDoesntExist # Compliant
def ok1():
    pass

@unrelatedDecorator.exempt
def ok2():
    pass
