from flask import Flask, Blueprint, jsonify, render_template, render_template_string, make_response, Response

# ----------------------------------------------------------------------------
# Test cases: error handlers without explicit status codes
# ----------------------------------------------------------------------------

def render_template_without_status():
    app = Flask(__name__)

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html')  # Noncompliant {{Specify an explicit HTTP status code for this error handler.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def jsonify_without_status():
    app = Flask(__name__)

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify(error="Internal server error")  # Noncompliant {{Specify an explicit HTTP status code for this error handler.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def render_template_string_without_status():
    app = Flask(__name__)

    @app.errorhandler(500)
    def internal_error(e):
        return render_template_string("<h1>Error</h1>")  # Noncompliant

def string_without_status():
    app = Flask(__name__)

    @app.errorhandler(400)
    def bad_request(e):
        return 'Bad request!'  # Noncompliant {{Specify an explicit HTTP status code for this error handler.}}
#       ^^^^^^^^^^^^^^^^^^^^^

def dict_without_status():
    app = Flask(__name__)

    @app.errorhandler(403)
    def forbidden(e):
        return {'error': 'Forbidden'}  # Noncompliant
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def blueprint_errorhandler_without_status():
    bp = Blueprint('main', __name__)

    @bp.errorhandler(404)
    def bp_not_found(e):
        return jsonify(error="Not found")  # Noncompliant
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def blueprint_app_errorhandler_without_status():
    bp = Blueprint('api', __name__)

    @bp.app_errorhandler(500)
    def api_error(e):
        return jsonify(error="API error")  # Noncompliant
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def multiple_returns_without_status():
    app = Flask(__name__)

    @app.errorhandler(404)
    def not_found(e):
        if e.description:
            return jsonify(error=e.description)  # Noncompliant
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        return jsonify(error="Not found")  # Noncompliant
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# ----------------------------------------------------------------------------
# Test cases: error handlers with explicit status codes
# ----------------------------------------------------------------------------

def tuple_return_with_status():
    app = Flask(__name__)

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404

def tuple_with_three_elements():
    app = Flask(__name__)

    @app.errorhandler(403)
    def forbidden(e):
        return jsonify(error='Forbidden'), 403, {'X-Error': 'Access denied'}

def make_response_with_status_arg():
    app = Flask(__name__)

    @app.errorhandler(404)
    def not_found(e):
        return make_response("Not found", 404)

def response_with_status_kwarg():
    app = Flask(__name__)

    @app.errorhandler(500)
    def server_error(e):
        return Response("Server error", status=500)

def variable_assigned_tuple_with_status():
    app = Flask(__name__)

    @app.errorhandler(500)
    def internal_error(e):
        result = jsonify(error="Error"), 500
        return result

def status_code_set_via_attribute():
    app = Flask(__name__)

    @app.errorhandler(404)
    def not_found(e):
        response = make_response("Not found")
        response.status_code = 404
        return response

def unknown_function_call():
    app = Flask(__name__)

    @app.errorhandler(500)
    def server_error(e):
        # Arbitrary function calls are not analyzed
        return build_error_response("Server error", 500)

# ----------------------------------------------------------------------------
# Edge cases - boundary conditions and special scenarios
# ----------------------------------------------------------------------------

def empty_return():
    app = Flask(__name__)

    @app.errorhandler(404)
    def page_not_found(e):
        return  # Noncompliant {{Specify an explicit HTTP status code for this error handler.}}
#       ^^^^^^

def variable_assigned_from_problematic_function():
    app = Flask(__name__)

    @app.errorhandler(500)
    def server_error(e):
        error_response = jsonify(error="Server error")
        return error_response  # Noncompliant
#       ^^^^^^^^^^^^^^^^^^^^^

def status_code_read_but_not_set():
    app = Flask(__name__)

    @app.errorhandler(500)
    def server_error(e):
        response = jsonify(error="Server error")
        print(response.status_code)
        return response  # Noncompliant
#       ^^^^^^^^^^^^^^^

def variable_assigned_tuple():
    app = Flask(__name__)

    @app.errorhandler(404)
    def not_found(e):
        response = ("Not found", 404)
        return response

def nested_function_not_checked():
    app = Flask(__name__)

    @app.errorhandler(500)
    def outer(e):
        def nested():
            # Nested functions are not visited
            return jsonify(error="nested error")
        return jsonify(error="Error"), 500

def decorator_not_call_expression():
    app = Flask(__name__)

    # @app.errorhandler without parentheses is not a CallExpression
    @app.errorhandler
    def page_not_found(e):
        return "Not found"

def chained_name_assignment():
    app = Flask(__name__)

    @app.errorhandler(500)
    def page_not_found(e):
        a = ("error", 404)
        b = a
        return b

def unknown_return():
    app = Flask(__name__)

    @app.errorhandler(500)
    def page_not_found(e):
        return unknown


def unknown_return():
    app = Flask(__name__)

    def some_function():
        ...

    @app.errorhandler(500)
    def page_not_found(e):
        return some_function


def not_error_handler():
    app = Flask(__name__)

    def regular_function():
        # Not decorated with @app.errorhandler - not checked
        return jsonify(data="some data")
