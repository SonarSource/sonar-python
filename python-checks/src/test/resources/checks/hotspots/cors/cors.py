# ------------- DJANGO ---------------------------

CORS_ORIGIN_ALLOW_ALL = True # ok, not in Django "settings.py"
class MyResponse:
    pass

def django_response():
    from django.http import HttpResponse
    response = HttpResponse("OK")
    response["Access-Control-Allow-Origin"] = "*"   # Noncompliant {{Make sure this permissive CORS policy is safe here.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    response["Access-Control-Allow-Origin"] = "trustedwebsite.com" # Compliant
    response[42] = "*" # Compliant
    foo.response["Access-Control-Allow-Origin"] = "*" # Compliant
    response["Access-Control-Allow-Credentials"] = "" # Compliant
    response["Access-Control-Expose-Headers"] = "" # Compliant
    response["Access-Control-Max-Age"] = "" # Compliant
    response["Access-Control-Allow-Methods"] = "" # Compliant
    response["Access-Control-Allow-Headers"] = "" # Compliant
    response.__setitem__("Access-Control-Allow-Origin", "*") # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    response.__setitem__("Access-Control-Allow-Methods", "*") # OK
    response.__setitem__("Access-Control-Allow-Origin", "foobar") # OK
    response.__setitem__("Access-Control-Allow-Origin") # OK
    response.foo("Access-Control-Allow-Origin", "*") # OK

    from django.http.response import HttpResponse as HttpResponseAltImport
    response_alt_import = HttpResponseAltImport("OK")
    response_alt_import["Access-Control-Allow-Origin"] = "*" # Noncompliant

    responseUnknown = UnknownResponse()
    responseUnknown["Access-Control-Allow-Origin"] = "*"

    a, b = 1, 2
    r = MyResponse()
    r["Access-Control-Allow-Origin", 42] = "*"
    r["Access-Control-Allow-Origin"] = "*"

# ------------- FLASK ---------------------------

import flask

def flask_cors():
    from flask_cors import CORS
    app = flask.Flask(__name__)
    CORS(app) # FN Noncompliant
    CORS(app, origins="*") # FN Noncompliant
    CORS(app, origins=r".*") # FN Noncompliant
    CORS(app, origins=r".+") # FN Noncompliant
    CORS(app, origins=r"^.*$") # FN Noncompliant
    CORS(app, origins=r"^.+$") # FN Noncompliant
    CORS(app, origins=r"^.+") # FN Noncompliant
    CORS(app, origins=["*"]) # FN Noncompliant
    CORS(app, origins="trustedwebsite.com") # Compliant
    CORS(app, origins=0) # Compliant
    CORS(app, origins=["trustedwebsite.com"]) # Compliant
    CORS(app, resources=r"/api/*") # FN Noncompliant
    CORS(app, resources=r"/api/*", origins="trustedwebsite.com") # OK
    CORS(app, resources=0) # OK
    CORS(app, resources={r"/api/1": {"origins": "*"}, r"/api/2": {"origins": "*"}}) # FN Noncompliant
    CORS(app, resources={r"/api/*": {"origins": r".*"}}) # FN Noncompliant
    CORS(app, resources={r"/api/*": {"origins": r".*", "something_else": 42}}) # FN Noncompliant
    CORS(app, resources={r"/api/*": {"origins": r".+"}}) # FN Noncompliant
    CORS(app, resources={r"/api/*": {"origins": ["*"]}}) # FN Noncompliant
    CORS(app, resources={r"/api/*": {"foo": ["*"]}}) # OK
    CORS(app, resources={r"/api/*": {"origins": "trustedwebsite.com"}, **unpack}) # Compliant
    CORS(app, resources={r"/api/*": {"origins": ["trustedwebsite.com"]}}) # Compliant

def flask_cross_origin_decorator():
    from flask_cors import cross_origin
    @cross_origin() # FN Noncompliant
##   ^^^^^^^^^^^^^^^
    @cross_origin # FN Noncompliant
##   ^^^^^^^^^^^^^
    @cross_origin(origins="*") # FN Noncompliant
##   ^^^^^^^^^^^^^^^^^^^^^^^^^^
    @cross_origin(origins=r".*") # FN Noncompliant
    @cross_origin(origins=r".+") # FN Noncompliant
    @cross_origin(origins=["*"]) # FN Noncompliant
    @cross_origin(origins="trustedwebsite.com") # Compliant
    @cross_origin(origins=["trustedwebsite.com"]) # Compliant
    @foo.cross_origin() # compliant
    @foo_cross_origin() # compliant
    def foo():
        pass

def flask_response_headers():
    flask.Response("{}", 200, {"Access-Control-Allow-Origin": "*"}) # Noncompliant
#                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    flask.wrappers.Response("{}", 200, {"Access-Control-Allow-Origin": "*"}) # Noncompliant
    flask.make_response(("{}", 200, {"Access-Control-Allow-Origin": "*"})) # Noncompliant
    return flask.helpers.make_response(("{}", {"Access-Control-Allow-Origin": "*"})) # Noncompliant

def werkzeug_headers():
    from werkzeug.datastructures import Headers
    Headers({"Access-Control-Allow-Origin": "*"}) # Noncompliant
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    headers = Headers({"Access-Control-Allow-Origin": "trustedwebsite.com"}) # Compliant
    headers.set("Access-Control-Allow-Origin", "*") # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    headers.setdefault("Access-Control-Allow-Origin", "*") # Noncompliant
    headers["Access-Control-Allow-Origin"] = "*" # Noncompliant
    headers.__setitem__("Access-Control-Allow-Origin", "*") # Noncompliant

    headers.set("Access-Control-Allow-Credentials", "") # Compliant
    headers.set("Access-Control-Expose-Headers", "") # Compliant
    headers.set("Access-Control-Max-Age", "") # Compliant
    headers.set("Access-Control-Allow-Methods", "") # Compliant
    headers.set("Access-Control-Allow-Headers", "") # Compliant
    Headers(1, 2)

from flask import Flask

app = Flask()

@app.after_request
def flask_add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*') # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    my.response.headers.add('Access-Control-Allow-Origin', '*')
    response.non_headers.add('Access-Control-Allow-Origin', '*')


def flask_add_headers_nullable_arg(a, /):
    response.headers.add('Access-Control-Allow-Origin', '*')

def flask_add_headers_not_a_first_param(response, secondparam):
    secondparam.headers.add('Access-Control-Allow-Origin', '*')

def flask_add_headers_no_args():
    response.headers.add('Access-Control-Allow-Origin', '*')

# not defined in a function
response.headers.add('Access-Control-Allow-Origin', '*')

from flask import make_response

@app.route("/")
def flask_make_response_headers_set():
    resp = make_response("hello")
    resp.headers['Access-Control-Allow-Origin'] = '*' # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return resp


@app.route("/")
def flask_app_response_headers_set():
    resp = app.Response("Foo bar baz")
    resp.headers['Access-Control-Allow-Origin'] = '*' # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    resp.non_headers['Access-Control-Allow-Origin'] = '*'
    resp.resp.headers['Access-Control-Allow-Origin'] = '*'
    unknown_response.headers['Access-Control-Allow-Origin'] = '*'
    non_response = 42
    non_response.headers['Access-Control-Allow-Origin'] = '*'
    non_call_response = app.NonResponse("Foo bar baz")
    non_call_response.headers['Access-Control-Allow-Origin'] = '*'
    return resp
