from flask import Flask, Blueprint, request

app = Flask(__name__)
bp = Blueprint('bp', __name__)


def noncompliant_examples():
    @app.route('/api/users')  # Noncompliant {{Specify the HTTP methods this route should accept.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^
    def handle_users_post():
        if request.method == 'POST':
            return 'created'
        return 'users'


    @app.route('/dashboard')  # Noncompliant
    def dashboard():
        return 'dashboard'



def compliant_examples():
    @app.route('/dashboard', methods=['GET'])
    def explicit_get_method():
        return 'dashboard'


    @app.route('/api/users', methods=['GET', 'POST'])
    def handle_users_with_methods():
        if request.method == 'POST':
            return 'created'
        return 'users'


    @app.route
    def decorator_without_call():
        if request.method == 'POST':
            return 'post'
        return 'get'

    @app.route('/dynamic', methods=get_methods())
    def dynamic_methods():
        if request.method == 'POST':
            return 'post'
        return 'get'


    @other_decorator('/path')
    def not_flask_route():
        if request.method == 'POST':
            return 'not flask'
        return 'data'


def edge_cases():
    # Dictionary unpacking - we cannot determine if methods is present, so don't raise
    @app.route('/unpacked', **kwargs)
    def unpacked_args():
        if request.method == 'POST':
            return 'post'
        return 'get'

    @app.route('/methods_string', methods='POST')  # methods as string, not list
    def methods_as_string():
        if request.method == 'POST':
            return 'post'
        return 'get'

    @app.route('/methods_tuple', methods=('GET', 'POST'))  # methods as tuple
    def methods_as_tuple():
        if request.method == 'POST':
            return 'post'
        return 'get'

    # Dictionary unpacking with literal dict
    @app.route('/literal_unpack', **{'methods': ['GET', 'POST']})
    def literal_dict_unpack():
        if request.method == 'POST':
            return 'post'
        return 'get'

    # Dictionary unpacking with variable
    options = {'methods': ['GET', 'POST']}
    @app.route('/variable_unpack', **options)
    def variable_dict_unpack():
        if request.method == 'POST':
            return 'post'
        return 'get'
