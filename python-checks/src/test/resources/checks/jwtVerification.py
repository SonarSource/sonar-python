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

def peek_then_verify_options_dict(token, keys):
    unverified = jwt.decode(token, options={"verify_signature": False, "require": ["iss"]}) # Compliant: same token is re-decoded below with real verification
    issuer = unverified["iss"]
    key = keys[issuer]
    return jwt.decode(token, key, algorithms=["HS256"])

def peek_then_verify_options_list(token, keys):
    unverified = jwt.decode(token, options=[("verify_signature", False), ("require", ["iss"])]) # Compliant: same token is re-decoded below with real verification
    issuer = unverified["iss"]
    key = keys[issuer]
    return jwt.decode(token, key, algorithms=["HS256"])

def peek_then_verify_verify_kwarg(token, keys):
    unverified = jwt.decode(token, verify=False) # Compliant: same token is re-decoded below with real verification
    issuer = unverified["iss"]
    key = keys[issuer]
    return jwt.decode(token, key, algorithms=["HS256"])

# only token_b is re-verified below, not token_a
def peek_then_verify_different_variable(token_a, token_b, keys):
    unverified = jwt.decode(token_a, options={"verify_signature": False}) # Noncompliant
    key = keys[unverified["iss"]]
    return jwt.decode(token_b, key, algorithms=["HS256"])

# the later call on the same token is also unverified
def peek_then_verify_second_call_also_unverified(token, keys):
    unverified = jwt.decode(token, options={"verify_signature": False}) # Noncompliant
    issuer = unverified["iss"]
    key = keys[issuer]
    return jwt.decode(token, key, algorithms=["HS256"], verify=False) # Noncompliant

def peek_then_verify_unresolvable_token(keys):
    return jwt.decode(get_token(), options={"verify_signature": False}) # Compliant: token argument isn't a resolvable variable, assume compliant

# real verification below is in a different function, out of scope
def peek_then_verify_wrong_scope(token, keys):
    def inner():
        return jwt.decode(token, options={"verify_signature": False}) # Noncompliant
    def other():
        return jwt.decode(token, keys["k"], algorithms=["HS256"])
    return inner(), other()

module_token = "..."
module_key = "secret"
module_unverified = jwt.decode(module_token, options={"verify_signature": False}) # Compliant: real verification happens at module scope below
module_verified = jwt.decode(module_token, module_key, algorithms=["HS256"])

def pyjwt_decode_unverified_header(token):
    return jwt.get_unverified_header(token) # Noncompliant

def get_unverified_header_access(token:str):
    header = jwt.get_unverified_header(token)  # Noncompliant
    print(f"Extra data in header: {header['extra']}")

def get_unverified_header_return(token: str) -> Dict[str, str]:
    header = jwt.get_unverified_header(token)  # Noncompliant
    return header

def get_unverified_header_non_compliant_sanity_check(token: str, some_object, other_call) -> Dict[str, str]:
    other = header = jwt.get_unverified_header(token)  # Noncompliant
    other.get("kid")

    some_object[0] = jwt.get_unverified_header(token)  # Noncompliant
    header = jwt.get_unverified_header(token)  # Noncompliant

def get_unverified_header_sanity_checks(token: str , other_call) -> Dict[str, str]:
    header = jwt.get_unverified_header(token)  # Noncompliant
    header.test("kid")
    header.get
    header.get()
    header[slice(12)]
    other_call(jwt.get_unverified_header(token).get("x5u"))
    return jwt.get_unverified_header(token).get("kid")

def get_unverified_header_used(token: str, do_other_things_with):
    header = jwt.get_unverified_header(token)  # Noncompliant
    return do_other_things_with(header)

def get_unverified_header_disallowed_access(token: str):
    header = jwt.get_unverified_header(token)  # Noncompliant
    kid = header.get("kid")
    not_kid = header.get("extra")

    header = jwt.get_unverified_header(token)  # Noncompliant
    kid = header.get("kid")
    not_kid = header["extra"]

def get_unverified_header_compliant(token: str, keys):
    header = jwt.get_unverified_header(token)  # Compliant: only "kid" is accessed
    kid = header.get("kid")

    x5u = jwt.get_unverified_header(token).get("x5u")  # Compliant

    x5t  = jwt.get_unverified_header(token)["x5t"]  # Compliant
    header = jwt.get_unverified_header(token) # Compliant
    jku = header["jku"]
    key = keys[jku]
    claims = jwt.decode(token, key, algorithms=["HS256"])
    return claims

def header_comparison_whole_header(token: str):
    expected_header = {"alg": "RS256", "typ": "JWT"}
    received_header = jwt.get_unverified_header(token)  # Compliant: only compared, never used for a decision
    if expected_header != received_header:
        raise ValueError("Invalid header")
    return True

def header_comparison_inline_no_assignment(token: str, expected):
    if jwt.get_unverified_header(token) == expected:  # Compliant: compared inline, never assigned
        return True
    return False

def header_comparison_extracted_value_inline_no_assignment(token: str):
    if jwt.get_unverified_header(token).get("alg") != "RS256":  # Compliant: extracted value compared inline
        raise ValueError("Wrong algorithm")
    return True

def header_comparison_subscription_inline_no_assignment(token: str):
    if jwt.get_unverified_header(token)["alg"] != "RS256":  # Compliant: extracted value compared inline
        raise ValueError("Wrong algorithm")
    return True

def header_comparison_extracted_value(token: str):
    received_header = jwt.get_unverified_header(token)  # Compliant: extracted value only compared
    if received_header.get("alg") != "RS256":
        raise ValueError("Wrong algorithm")
    return True

def header_comparison_subscription(token: str):
    received_header = jwt.get_unverified_header(token)  # Compliant: extracted value only compared
    if received_header["alg"] != "RS256":
        raise ValueError("Wrong algorithm")
    return True

def header_comparison_multiple_safe_usages(token: str):
    expected_header = {"alg": "RS256", "typ": "JWT"}
    received_header = jwt.get_unverified_header(token)  # Compliant: every usage is a safe comparison
    if expected_header != received_header:
        raise ValueError("Invalid header")
    if received_header == expected_header:
        return True
    if received_header.get("alg") != "RS256":
        raise ValueError("Wrong algorithm")
    return True


# header is also returned raw below - comparing it once doesn't make the other, unsafe usage safe
def header_comparison_mixed_with_unsafe_use(token: str) -> Dict[str, str]:
    header = jwt.get_unverified_header(token)  # Noncompliant
    if header.get("alg") != "RS256":
        raise ValueError("Wrong algorithm")
    return header

def algorithm_validated_then_used(token: str, key):
    header = jwt.get_unverified_header(token)  # Compliant: alg is validated against an allowlist before use
    alg = header.get("alg")
    if alg not in ["RS256", "ES256"]:
        raise ValueError("Unsupported algorithm")
    return jwt.decode(token, key, algorithms=[alg])

def algorithm_validated_then_used_subscription(token: str, key):
    header = jwt.get_unverified_header(token)  # Compliant: alg is validated against an allowlist before use
    alg = header["alg"]
    if alg not in ("RS256", "ES256"):
        raise ValueError("Unsupported algorithm")
    return jwt.decode(token, key, algorithms=[alg])

# alg is used to decode BEFORE it's validated - the guard doesn't protect this decode call
def algorithm_used_before_validated(token: str, key):
    header = jwt.get_unverified_header(token)  # Noncompliant
    alg = header.get("alg")
    payload = jwt.decode(token, key, algorithms=[alg])
    if alg not in ["RS256", "ES256"]:
        raise ValueError("Unsupported algorithm")
    return payload


# alg is used directly in algorithms=[...] without ever being validated against an allowlist
def algorithm_used_without_validation(token: str, key):
    header = jwt.get_unverified_header(token)  # Noncompliant
    alg = header.get("alg")
    return jwt.decode(token, key, algorithms=[alg])

def validate_alg_elsewhere(alg):
    if alg not in ["RS256"]:
        raise ValueError("Unsupported algorithm")

# the allowlist guard lives in a sibling function, out of scope for this one
def algorithm_validated_wrong_scope(token: str, key):
    header = jwt.get_unverified_header(token)  # Noncompliant
    alg = header.get("alg")
    return jwt.decode(token, key, algorithms=[alg])

def algorithm_validated_then_used_bare_name(token: str, key):
    header = jwt.get_unverified_header(token)  # Compliant: alg is validated before being passed as a bare algorithms= value
    alg = header.get("alg")
    if alg not in ["RS256", "ES256"]:
        raise ValueError("Unsupported algorithm")
    return jwt.decode(token, key, algorithms=alg)

# alg is never bound to a simple name (used inline), so it can't be tracked for validation
def algorithm_extracted_inline_no_validation(token: str, key):
    header = jwt.get_unverified_header(token)  # Noncompliant
    return jwt.decode(token, key, algorithms=[header.get("alg")])
