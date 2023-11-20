from fastapi import APIRouter
import json
from jose import jwt
from jose import jws
from jose.exceptions import JWTError, ExpiredSignatureError, JWTClaimsError

router = APIRouter()


@router.get('/noncompliant_jwt_get_unverified_header')
async def noncompliant_jwt_get_unverified_header(token: str | None = None):
    try:
        return jwt.get_unverified_header(token)  # Noncompliant {{Don't use a JWT token without verifying its signature.}}
    #                                    ^^^^^
    finally:
        ...


@router.get('/noncompliant_jwt_get_unverified_headers')
async def noncompliant_jwt_get_unverified_headers(token: str | None = None):
    try:
        return jwt.get_unverified_headers(token)  # Noncompliant
    #                                     ^^^^^
    finally:
        ...


@router.get('/noncompliant_jws_get_unverified_header')
async def noncompliant_jws_get_unverified_header(token: str | None = None):
    try:
        return jws.get_unverified_header(token)  # Noncompliant
    #                                    ^^^^^
    finally:
        ...


@router.get('/noncompliant_jws_get_unverified_headers')
async def noncompliant_jws_get_unverified_headers(token: str | None = None):
    try:
        return jws.get_unverified_headers(token)  # Noncompliant
    #                                     ^^^^^
    finally:
        ...


@router.get('/noncompliant_jwt_get_unverified_claims')
async def noncompliant_jwt_get_unverified_claims(token: str | None = None):
    try:
        return jwt.get_unverified_claims(token)  # Noncompliant
    #                                    ^^^^^
    finally:
        ...


@router.get('/noncompliant_jws_get_unverified_claims')
async def noncompliant_jws_get_unverified_claims(token: str | None = None):
    try:
        payload = jws.get_unverified_claims(token)  # Noncompliant
    #                                       ^^^^^
        ...
    finally:
        ...


@router.get('/noncompliant_decode_dict_1')
async def noncompliant_decode_dict_1(token: str | None = None):
    try:
        return jwt.decode(token, None, options={"verify_signature": False})  # Noncompliant {{Don't use a JWT token without verifying its signature.}}
    #                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    finally:
        ...


@router.get('/noncompliant_decode_dict_2')
async def noncompliant_decode_dict_2(token: str | None = None):
    options = {"verify_signature": False}
    try:
        return jwt.decode(token, None, options=options)  # Noncompliant
    #                                          ^^^^^^^
    finally:
        ...


@router.get('/noncompliant_decode_dict_3')
async def noncompliant_decode_dict_3(token: str | None = None):
    options = dict(verify_signature=False)
    try:
        return jwt.decode(token, None, options=options)  # Noncompliant
    #                                          ^^^^^^^
    finally:
        ...


@router.get('/noncompliant_list_1')
async def noncompliant_list_1(token: str | None = None):
    try:
        return jwt.decode(token, None, options=[("leeway", 10), ("verify_signature", False)])  # Noncompliant
    #                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    finally:
        ...


@router.get('/noncompliant_list_2')
async def noncompliant_list_2(token: str | None = None):
    options = [("leeway", 10), ("verify_signature", False)]
    try:
        return jwt.decode(token, None, options=options)  # Noncompliant
    #                                          ^^^^^^^
    finally:
        ...



@router.get('/noncompliant_verify')
async def noncompliant_verify(token: str | None = None):
    try:
        payload = jws.verify(token, None, None, verify=False)  # Noncompliant
    #                                           ^^^^^^^^^^^^
        ...
    finally:
        ...


@router.get('/compliant_decode_dict_1')
async def compliant_decode_dict_1(token: str | None = None):
    something: int
    try:
        return jwt.decode(token, None, options={"verify_signature": True, something: something})
    finally:
        ...


@router.get('/compliant_decode_dict_2')
async def compliant_decode_dict_2(token: str | None = None):
    options = {"verify_signature": True}
    try:
        return jwt.decode(token, None, options=options)
    finally:
        ...


@router.get('/compliant_decode_dict_3')
async def compliant_decode_dict_3(token: str | None = None):
    options = dict(verify_signature=True)
    try:
        return jwt.decode(token, None, options=options)
    finally:
        ...

@router.get('/compliant_decode_dict_4')
async def compliant_decode_dict_4(token: str | None = None):
    try:
        return jwt.decode(token, None, options=dict(verify_signature=True))
    finally:
        ...


@router.get('/compliant_list_1')
async def compliant_list_1(token: str | None = None):
    try:
        return jwt.decode(token, None, options=[("leeway", 10), ("verify_signature", True), ("e1", "e2", "e3")])
    finally:
        ...


@router.get('/compliant_list_2')
async def compliant_list_2(token: str | None = None):
    options = [("leeway", 10), ("verify_signature", True)]
    try:
        return jwt.decode(token, None, options=options)
    finally:
        ...


def some_function():
    ...


@router.get('/compliant')
async def compliant(token: str | None = None):
    JWT_SIGNING_KEY = some_function()
    try:
        return jwt.decode(token, JWT_SIGNING_KEY, algorithms=['HS256'])
    finally:
        ...

@router.get('/compliant_decode_arbitrary_function')
async def compliant_decode_arbitrary_function(token: str | None = None):
    try:
        return jwt.decode(token, None, options=some_function(verify_signature=True))
    finally:
        ...

@router.get('/compliant_decode_arbitrary_function')
async def compliant_decode_arbitrary_function(token: str | None = None):
    x = True
    x = False
    try:
        return jwt.decode(token, None, options=x)
    finally:
        ...

@router.get('/compliant_decode_string_option')
async def compliant_decode_string_option(token: str | None = None):
    args = ("arg1", "arg2")
    try:
        return jwt.decode(token, None, options="some_string")
    finally:
        ...

@router.get('/compliant_verify')
async def compliant_verify(token: str | None = None):
    try:
        payload = jws.verify(token, None, None, verify=True)
        ...
    finally:
        ...


x = []
class TestInfiniteRecursion():
    x = x
    payload = jwt.decode(token, None, options=x)
