from flask import Flask, request, Blueprint

app = Flask(__name__)
bp = Blueprint('bp', __name__)


@app.route('/resource', methods=['POST'])
def update_text():
    key = request.args.get('key')  # Noncompliant {{Do not use query parameters with POST requests; use path parameters or request body instead.}}
    #     ^^^^^^^^^^^^
    data = request.get_data()
    return 'Updated'

@app.route('/user', methods=['POST'])
def create_user():
    user_id = request.args['user_id']  # Noncompliant
    #         ^^^^^^^^^^^^
    name = request.get_json().get('name')
    return f'User {user_id} created'

@app.route('/item', methods=['POST', 'PUT'])
def update_item():
    item_id = request.args.get('id')  # Noncompliant
    #         ^^^^^^^^^^^^
    return 'Updated'

@app.route('/data', methods=['post'])  # lowercase
def update_data():
    value = request.args.get('value')  # Noncompliant
    #       ^^^^^^^^^^^^
    return 'OK'

@app.route('/data', methods=['POST'])
def getting_args():
    value = request.args  # Noncompliant
    #       ^^^^^^^^^^^^
    return value

@bp.route('/resource', methods=['POST', 'PUT'])
def update_resource():
    key = request.args.get('key')  # Noncompliant
    #     ^^^^^^^^^^^^
    response.set_cookie(request.args.get("name"), request.args.get("value"))  # Noncompliant 2
    return 'Updated'

@app.route('/data', methods=['GET','POST'])
def get_and_post():
    value = request.args.get("key") # Compliant: could be the GET route
    return value

@app.route('/data', methods=['OPTIONS','POST'])
def option_and_post():
    value = request.args.get("key") # Compliant: could be the OPTIONS route
    return value

@app.route('/resource', methods=['GET'])
def get_resource():
    key = request.args.get('key')  # Compliant: GET method
    return f'Resource {key}'

@app.route('/data', methods=['POST'])
def post_data():
    data = request.get_json()
    body = request.get_data()
    form = request.form.get('field')
    return 'OK'

@app.route('/item')  # No methods specified, defaults to GET
def get_item():
    item_id = request.args.get('id')  
    return f'Item {item_id}'

@app.route('/update', methods=['PUT', 'DELETE'])
def update_or_delete():
    key = request.args.get('key')  
    return 'Updated'


@app.route
def coverage_decorator():
    key = request.args.get('key')  # Compliant: decorator is not a call expression
    return key

def regular_function(*args):
    key = request.args.get('key')  # Compliant: not a route
    return key

@regular_function('/update', methods=['PUT', 'DELETE'])
def not_flask_route():
    key = request.args.get('key')
    return key

@app.route('/update', methods='POST')
def incorrect_methods_object():
    key = request.args.get('key')  
    return 'Updated'

@app.route('/update', methods=['POST'])
def request_coverage():
    key = request().args.get('key')  
    return 'Updated'
