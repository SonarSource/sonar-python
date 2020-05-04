from flask import Flask
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
app.config['SECRET_KEY'] = 'top-secret!'
csrf = CSRFProtect()
csrf.init_app(app) # Compliant

@app.route('/csrftest1/', methods=['POST'])
@csrf.exempt # Noncompliant {{Disabling CSRF protection is dangerous.}}
#     ^^^^^^
def csrftestpost():
    pass
