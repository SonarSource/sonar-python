class Methods:
    def route(self): ...
def foo(): ...

methods = Methods()

@methods.route('/sensitive1', methods=['GET', 'POST'])  # OK
def compliant():
    return Response("Method: " + request.method, 200)

@foo('/sensitive1', methods=['GET', 'POST'])  # OK
def compliant2():
    return Response("Method: " + request.method, 200)
