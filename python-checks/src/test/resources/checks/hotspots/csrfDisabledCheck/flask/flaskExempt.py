def exemptExample():
  from flask import Flask
  from flask_wtf.csrf import CSRFProtect
  from exportedCsrfProtect import csrfProtect as exported_csrf_protect
  
  app = Flask(__name__)
  csrfProtect = CSRFProtect()
  csrfProtect.init_app(app) # Compliant

  @app.route('/csrftest1/', methods=['POST'])
  @csrfProtect.exempt # Noncompliant {{Make sure disabling CSRF protection is safe here.}}
  #            ^^^^^^
  def csrftestpost():
      pass

  @app.route('/csrftest1/', methods=['POST'])
  @exported_csrf_protect.exempt # Noncompliant
  def csrftestpost_rexported():
      pass
