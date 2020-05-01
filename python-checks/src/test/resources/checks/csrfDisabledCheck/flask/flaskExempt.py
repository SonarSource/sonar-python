def exemptExample():
  from flask import Flask
  from flask_wtf.csrf import CSRFProtect
  
  app = Flask(__name__)
  csrfProtect = CSRFProtect()
  csrfProtect.init_app(app) # Compliant

  @app.route('/csrftest1/', methods=['POST'])
  @csrfProtect.exempt # Noncompliant {{Disabling CSRF protection is dangerous.}}
  #            ^^^^^^
  def csrftestpost():
      pass
