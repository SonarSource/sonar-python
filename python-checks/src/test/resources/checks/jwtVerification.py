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

def pyjwt_decode_token_1(token):
    return jwt.decode(token, options={"verify_signature":False, "something":"something"}) # Noncompliant

def pyjwt_decode_token_2(token):
    return jwt.decode(token, options=[("verify_signature", False), ("something", "something")]) # Noncompliant

def pyjwt_decode_token_secure_1(token):
    return jwt.decode(token, algorithms="HS256", options={"verify_signature":True, "something":"something"}) # Compliant

def pyjwt_decode_token_secure_2(token):
    return jwt.decode(token, algorithms="HS256") # Compliant

def pyjwt_decode_unverified_header(token):
    return jwt.get_unverified_header(token) # Noncompliant

def get_unverified_header_access(token:str):
    header = jwt.get_unverified_header(token)  # Noncompliant
    print(f"Extra data in header: {header['extra']}")

def get_unverified_header_return(token: str) -> Dict[str, str]:
    header = jwt.get_unverified_header(token)  # Noncompliant
    return header

def get_unverified_header_used(token: str, do_other_things_with):
    header = jwt.get_unverified_header(token)  # Noncompliant
    return do_other_things_with(header)

def get_unverified_header_disallowed_access(token: str, keys):
    header = jwt.get_unverified_header(token)  # Noncompliant
    kid = header.get("kid")
    not_kid = header.get("extra")

def get_unverified_header_compliant(token: str, keys):
    header = jwt.get_unverified_header(token)  # Compliant: only "kid" is accessed
    kid = header.get("kid")

    jwt.get_unverified_header(token).get("x5u")  # Compliant

    header = jwt.get_unverified_header(token) # Compliant
    jku = header["jku"]
    key = keys[jku]
    claims = jwt.decode(token, key, algorithms=["HS256"])
    return claims
