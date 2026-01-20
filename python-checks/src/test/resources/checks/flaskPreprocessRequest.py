from flask import Flask

app = Flask(__name__)

@app.before_request
def check_auth():
    # This is a before-request handler that may return a response
    pass

def handle_request():
    app.preprocess_request()  # Noncompliant {{Handle the return value of "preprocess_request()" to ensure before-request handlers' responses are not ignored.}}
#   ^^^^^^^^^^^^^^^^^^^^^^

    response = app.preprocess_request() #Compliant 
    if response is not None:
        return response

    return app.preprocess_request() #Compliant 

def no_issue_for_other_methods():
    app.run()
    app.test_client()

def no_issue_for_non_flask_preprocess_request():
    class CustomApp:
        def preprocess_request(self):
            pass

    custom_app = CustomApp()
    custom_app.preprocess_request() #Compliant 

def compliant_with_walrus_operator():
    if (response := app.preprocess_request()) is not None: #Compliant 
        return response

def expression_statement_without_call():
    app.name  # Compliant
