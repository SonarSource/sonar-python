def stdlib_tests(param):
    from http.cookies import SimpleCookie

    cookie = SimpleCookie()

    cookie['c1'] = 'value' # FN

    cookie['c2'] = 'value'
    cookie['c2']['secure'] = False # Noncompliant {{Make sure creating this cookie without the "secure" flag is safe.}}
    cookie['c2']['other'] = False # OK
    cookie['c2'][param] = False # OK

    cookie['c2']['secure', 'other'] = False # OK

    cookie['secure'] = False # OK

    cookie['c3'] = 'value'
    cookie['c3']['secure'] = True # OK

    not_a_cookie = 42
    not_a_cookie['c']['secure'] = False # OK

    get_cookie()['c']['secure'] = False # OK

def get_cookie(): pass

def django_tests():
    from django.http import HttpResponse
    from django.http import HttpResponseRedirect, HttpResponsePermanentRedirect, HttpResponseBadRequest, HttpResponseNotModified, HttpResponseNotFound, HttpResponseForbidden, HttpResponseNotAllowed, HttpResponseGone, HttpResponseServerError

    response = HttpResponse()
    response.set_cookie("C1", "world") # Noncompliant
    response.set_cookie("C2", "world", secure=None) # Noncompliant
    response.set_cookie("C3", "world", secure=False) # Noncompliant
    response.set_cookie("C4", "world", secure=True)

    response2 = HttpResponseRedirect()
    response2.set_cookie("C5", "world") # Noncompliant
    response2.set_cookie("C5", "world", None, None, "/", None, True) # OK
    response2.set_signed_cookie("C5", "world") # Noncompliant
    response2.set_signed_cookie("C5", "world", secure=True) # OK
    response2.set_signed_cookie("C5", "world", other=False, secure=True) # OK
    response2.set_signed_cookie("C5", "world", secure=False) # Noncompliant
    response2.set_signed_cookie("C5", "world", secure=None) # Noncompliant
    response2.set_signed_cookie("C5", "", "world", None, None, "/", None, True) # OK
    kwargs = { secure : True }
    response2.set_signed_cookie("C5", "world", **kwargs) # OK

    kwargs = { secure : False }
    response2.set_signed_cookie("C5", "world", **kwargs) # FN

    get_cookie().set_cookie("C3", "world", secure=False)

    response3 = HttpResponsePermanentRedirect()
    response3.set_cookie("C6", "world") # Noncompliant
    response4 = HttpResponseNotModified()
    response4.set_cookie("C7", "world") # Noncompliant
    response5 = HttpResponseBadRequest()
    response5.set_cookie("C8", "world") # Noncompliant
    response6 = HttpResponseNotFound()
    response6.set_cookie("C9", "world") # Noncompliant
    response7 = HttpResponseForbidden()
    response7.set_cookie("C10", "world") # Noncompliant
    response8 = HttpResponseNotAllowed()
    response8.set_cookie("C11", "world") # Noncompliant
    response9 = HttpResponseGone()
    response9.set_cookie("C12", "world") # Noncompliant
    response10 = HttpResponseServerError()
    response10.set_cookie("C13", "world") # Noncompliant

def flask_tests():
    import flask
    from flask import Response, make_response, redirect

    response1 = Response('OK')
    response1.set_cookie('c1', 'value') # Noncompliant

    response2 = flask.Response('OK')
    response2.set_cookie('c1', 'value', secure = True) # OK

    response3 = make_response()
    response3.set_cookie('c', 'value') # Noncompliant

    response4 = redirect()
    response4.set_cookie('c', 'value') # Noncompliant


from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/test1")
async def test1(response: Response):
    response.set_cookie("key", "value", secure=False)  # Noncompliant
    return {"message": "test"}


@app.get("/test2")
async def test2(response: Response):
    response.set_cookie("key", "value")  # Noncompliant
    return {"message": "test"}


@app.get("/test3")
async def test3(response: Response):
    response.set_cookie("key", "value", secure=True)  # Compliant
    return {"message": "test"}


@app.get("/test4")
async def test4():
    response = Response("test")
    response.set_cookie("key", "value")  # Noncompliant
    return response


@app.get("/test5")
async def test5():
    response = StreamingResponse(iter("test"), headers={"set-cookie": "key=value"})  # Noncompliant
    response = StreamingResponse(iter("test"), headers={"Set-Cookie": "key=value"})  # Noncompliant
    headers = {"Set-Cookie": "key=value"}
    response = StreamingResponse(iter("test"), headers=headers)  # Noncompliant
    response = StreamingResponse(iter("test"), headers=foo())
    return response

@app.get("/test6")
async def test6():
    response = StreamingResponse(iter("test"), headers={"Set-Cookie": "key=value; Secure"})  # OK
    return response

@app.get("/test7")
async def test7():
    response = StreamingResponse(iter("test"))  # OK
    return response
