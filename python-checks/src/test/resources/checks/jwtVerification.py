import jwt
import python_jwt

def jwt_decode_argument(token):
    jwt.decode(token)
    jwt.decode(token, verify = True)
    jwt.decode(token, verify = False) # Noncompliant
#                     ^^^^^^^^^^^^^^
    jwt.decode(token, verify = 42)
    jwt.decode(token, xx = False)
    jwt.decode(token, False)
    jwt.xxx(token, verify = False)
    xxx(token, verify = False)

def jwt_process_without_verify(token):
    jwt.process_jwt(token) # Noncompliant
    print(token)

def jwt_process_with_verify(token):
    jwt.process_jwt(token)
    jwt.verify_jwt(token)

def python_jwt_process_without_verify(token):
    python_jwt.process_jwt(token) # Noncompliant
    xxx(token)

def python_jwt_process_with_verify(token):
    python_jwt.process_jwt(token)
    python_jwt.verify_jwt(token)
