from flask import request, Response

def noncompliant_direct_access():
    auth_header = request.headers['Authorization']  # Noncompliant {{Use ".get()" method to safely access this header.}}
#                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def compliant_get_access():
    auth_header = request.headers.get('Authorization')
    user_agent = request.headers.get('User-Agent', 'Unknown')

def noncompliant_via_variable():
    headers = request.headers
    auth = headers['Authorization']  # Noncompliant

def compliant_unknown_type(req):
    return req.headers['SomeHeader']

def compliant_regular_dict():
    my_dict = {'key': 'value'}
    value = my_dict['key']

def compliant_iteration():
    for key in request.headers:
        print(key)

def compliant_setting_headers():
    response = Response()
    response.headers['Content-Type'] = 'application/json'
