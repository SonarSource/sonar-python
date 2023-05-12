def stdlib_tests(param):
    from http.cookies import SimpleCookie

    cookie = SimpleCookie()

    cookie['c1'] = 'value' # FN

    cookie['c2'] = 'value'
    cookie['c2']['httponly'] = False # Noncompliant {{Make sure creating this cookie without the "HttpOnly" flag is safe.}}
    cookie['c2']['other'] = False # OK
    cookie['c2'][param] = False # OK

    cookie['c2']['httponly', 'other'] = False # OK

    cookie['httponly'] = False # OK

    cookie['c3'] = 'value'
    cookie['c3']['httponly'] = True # OK

    not_a_cookie = 42
    not_a_cookie['c']['httponly'] = False # OK

    get_cookie()['c']['httponly'] = False # OK

def get_cookie(): pass

def django_tests():
    from django.http import HttpResponse
    from django.http import HttpResponseRedirect, HttpResponsePermanentRedirect, HttpResponseBadRequest, HttpResponseNotModified, HttpResponseNotFound, HttpResponseForbidden, HttpResponseNotAllowed, HttpResponseGone, HttpResponseServerError

    response = HttpResponse()
    response.set_cookie("C1", "world") # Noncompliant
    response.set_cookie("C2", "world", httponly=None) # Noncompliant
    response.set_cookie("C3", "world", httponly=False) # Noncompliant
    response.set_cookie("C4", "world", httponly=True)

    response2 = HttpResponseRedirect()
    response2.set_cookie("C5", "world") # Noncompliant
    response2.set_cookie("C5", "world", None, None, "/", None, True, True) # OK
    response2.set_signed_cookie("C5", "world") # Noncompliant
    response2.set_signed_cookie("C5", "world", httponly=True) # OK
    response2.set_signed_cookie("C5", "world", other=False, httponly=True) # OK
    response2.set_signed_cookie("C5", "world", httponly=False) # Noncompliant
    response2.set_signed_cookie("C5", "world", httponly=None) # Noncompliant
    response2.set_signed_cookie("C5", "", "world", None, None, "/", None, True, True) # OK
    kwargs = { httponly : True }
    response2.set_signed_cookie("C5", "world", **kwargs) # OK

    kwargs = { httponly : False }
    response2.set_signed_cookie("C5", "world", **kwargs) # FN

    get_cookie().set_cookie("C3", "world", httponly=False)

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
    response1.set_cookie('c1', 'value') # FN

    response2 = flask.Response('OK')
    response2.set_cookie('c1', 'value', httponly = True) # OK

    response3 = make_response()
    response3.set_cookie('c', 'value') # FN

    response4 = redirect()
    response4.set_cookie('c', 'value') # FN

def flask_SessionCookieHttpOnlyFalse():
    from flask import Flask, request, url_for, render_template, redirect, make_response, request, session

    app = Flask(__name__, static_url_path='/static', static_folder='static')

    app.config['DEBUG'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True # Ok
    app.config['SESSION_COOKIE_HTTPONLY'] = False # Noncompliant
    #                                       ^^^^^
    # Load default config and override config from an environment variable
    app.config.update({
        'SECRET_KEY': "woopie",
        'SESSION_COOKIE_HTTPONLY': False # Noncompliant
        #                          ^^^^^
    })
    app.config.update({
        'SECRET_KEY': "woopie",
        'SESSION_COOKIE_HTTPONLY': True # Ok
    })
    app.config.update(dict(
        SECRET_KEY = "woopie",
        SESSION_COOKIE_HTTPONLY = False # Noncompliant
        #                         ^^^^^
    ))
    app.config.update(dict(
        SECRET_KEY = "woopie",
        SESSION_COOKIE_HTTPONLY = True # Ok
    ))
    app.config.update(dict(
        42,
        **{'unpacking': 1},
        SECRET_KEY = "woopie",
        SESSION_COOKIE_HTTPONLY = True # Ok
    ))
