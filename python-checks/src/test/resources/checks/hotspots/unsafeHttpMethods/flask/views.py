from flask import Blueprint, request
from flask import Response
from exportedBlueprint import routes as other_routes

methods = Blueprint('methods', __name__)

# 127.0.0.1:5000/methods/sensitive1
@methods.route('/sensitive1', methods=['GET', 'POST'])  # Noncompliant
def sensitive1():
    return Response("Method: " + request.method, 200)

# 127.0.0.1:5000/methods/sensitive1
@other_routes.route('/sensitive1', methods=['GET', 'POST'])  # Noncompliant
def sensitive1_other_routes():
    return Response("Method: " + request.method, 200)

# 127.0.0.1:5000/methods/compliant1
@other_routes.route('/compliant1')  # Compliant
def compliant1_other_routes():
    return Response("Method: " + request.method, 200)

# 127.0.0.1:5000/methods/compliant1
@methods.route('/compliant1')  # Compliant
def compliant1():
    return Response("Method: " + request.method, 200)

# 127.0.0.1:5000/methods/compliant2
@methods.route('/compliant2', methods=['GET'])  # Compliant
def compliant2():
    return Response("Method: " + request.method, 200)

@methods.other('/compliant3', methods=['GET', 'POST'])  # Compliant
def compliant3():
    return Response("Method: " + request.method, 200)

@unknown('/compliant4', methods=['GET', 'POST'])  # Compliant
def compliant4():
    return Response("Method: " + request.method, 200)

@methods.route('/compliant5', methods=[getMethod()])  # Compliant
def compliant5():
    return Response("Method: " + request.method, 200)

@methods.route('/compliant6', methods=getMethods())  # Compliant
def compliant6():
    return Response("Method: " + request.method, 200)

@methods.route('/compliant7', methods=['FOO'])  # Compliant
def compliant7():
    return Response("Method: " + request.method, 200)

def inside_fn():
    from flask import Blueprint, request
    from flask import Response

    methods = Blueprint('methods', __name__)

    # 127.0.0.1:5000/methods/sensitive1
    @methods.route('/sensitive1', methods=['GET', 'POST'])  # Noncompliant
    def sensitive1():
        return Response("Method: " + request.method, 200)

    # 127.0.0.1:5000/methods/compliant1
    @methods.route('/compliant1')  # Compliant
    def compliant1():
        return Response("Method: " + request.method, 200)
