
def tests():
    # Reduced combination of
    #   flaskformcompliant1.py and
    #   flaskformsensitive1.py from
    # security-expected-issues/python/rules/hotspots/RSPEC-4502/flask
    from flask import Flask
    from flask import request
    from flask_wtf import Form, FlaskForm
    from wtforms import TextField
    from flask_wtf.csrf import CSRFProtect

    app = Flask(__name__)
    csrf = CSRFProtect(app) # it's not the point in this test. We are interested in the misconfigured forms here.

    # compliant examples
    class TestForm1(FlaskForm):
        name = TextField('name')

    class TestProtectedForm(FlaskForm):
        class Meta:
            csrf = True # Not needed but compliant

            somethingExceptCsrf = False # coverage only
            def onlyForCoverage():
                pass

    @app.route('/protectedflaskform1', methods=['GET', 'POST'])
    def protectedflaskform1():
        form = TestForm1(request.form) # Compliant by default

    @app.route('/protectedflaskform3', methods=['GET', 'POST'])
    def protectedflaskform3():
        form = TestProtectedForm(request.form, csrf_enabled=True) # Not needed but compliant

    # Sensitive examples
    class TestForm2(FlaskForm):
        name = TextField('name')

    class TestUnprotectedForm(FlaskForm):
        class Meta:
            csrf = False # Noncompliant {{Make sure disabling CSRF protection is safe here.}}
            #      ^^^^^

    @app.route('/unprotectedflaskform1', methods=['GET', 'POST'])
    def unprotectedflaskform1():
        form = TestForm2(request.form, csrf_enabled=False) # Noncompliant {{Make sure disabling CSRF protection is safe here.}}
        #                                           ^^^^^

    @app.route('/unprotectedflaskform2', methods=['GET', 'POST'])
    def unprotectedflaskform2():
        # This one wasn't originally marked as 'Sensitive', but spec suggested that it is
        form = TestForm2(request.form, meta={'csrf': False}) # Noncompliant {{Make sure disabling CSRF protection is safe here.}}
        #                                            ^^^^^

    @app.route('/unprotectedflaskform3', methods=['GET', 'POST'])
    def unprotectedflaskform3():
        form = TestUnprotectedForm(request.form)

    @app.route('/unprotectedflaskform4', methods=['GET', 'POST'])
    def unprotectedflaskform4():
        # code coverage corner cases
        form = TestForm2(request.form, meta={42: 58, 'foo': 'bar', 'csrf': False}) # Noncompliant {{Make sure disabling CSRF protection is safe here.}}
        #                                                                  ^^^^^

    # Corner cases for coverage
    class SomethingCompletelyUnrelated:
        class Meta:
            pass

    class Kwargform(FlaskForm):
        pass

    d = {}
    Kwargform(request.form, csrf_enabled=True, **d)

    thatsSomeUnknownMethod(app)
