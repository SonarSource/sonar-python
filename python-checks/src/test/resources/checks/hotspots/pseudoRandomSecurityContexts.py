import random
import string
from flask import make_response, Response
from django.http import HttpResponse
from starlette.responses import Response as ASGIResponse


def generate_captcha_challenge():
    return ''.join(random.choices('023456789', k=4))  # Noncompliant


def generate_captcha():
    first = random.randint(1, 100)  # Noncompliant
    second = random.randint(1, 100)  # Noncompliant
    operation = random.choice(['+', '-', '*'])  # Noncompliant
    return first, second, operation


class Channel:
    def _challenge(self):
        return '-'.join([str(random.randint(0, 0xfffffff)),  # Noncompliant
                         str(random.randint(0, 0xfffffff)),  # Noncompliant
                         str(random.randint(0, 0xfffffff))])  # Noncompliant

    def create(self):
        sid = '%08x' % random.randint(0, 1000000000)  # Noncompliant
        return sid


def render():
    response = make_response('body')
    response.set_cookie('session_id', ''.join(random.choice(string.ascii_uppercase) for _ in range(16)))  # Noncompliant
    return response


def render_named(response: Response):
    response.set_cookie(value=str(random.getrandbits(128)), key='session_id')  # Noncompliant
    response.set_cookie('csrf_token', str(random.getrandbits(128)))  # Noncompliant
    response.set_cookie(('sid'), (str(random.getrandbits(128))))  # Noncompliant


def render_other(response: HttpResponse, other: ASGIResponse):
    response.set_cookie('session_id', str(random.getrandbits(128)))  # Noncompliant
    other.set_cookie('session_id', str(random.getrandbits(128)))  # Noncompliant


def ordinary(response: Response):
    message = 'session_id captcha challenge sid'
    random.shuffle([1, 2, 3])
    response.set_cookie('theme', str(random.getrandbits(16)))
    response.set_cookie('session_id', 'fixed', max_age=random.randint(1, 100))
    response.set_cookie('session_id')
    response.set_cookie(str(random.getrandbits(16)), 'fixed')
    response.set_cookie('session_id', lambda: random.getrandbits(128))
    response.set_cookie('session_id', str(random.SystemRandom().getrandbits(128)))
    response.set_cookie(value=str(random.getrandbits(16)))
    response.set_cookie('session_id', 'fixed')
    return random.random()  # unrelated randomness in the same scope stays excluded


def unknown_receiver(response):
    response.set_cookie('session_id', str(random.getrandbits(128)))


def dynamic_name(response: Response, name):
    response.set_cookie(name, str(random.getrandbits(128)))


def ordinary_names():
    side = random.random()
    sidereal = random.random()
    challenged = random.random()
    return side, sidereal, challenged


def nested_scope():
    def challenge():
        return random.random()  # Noncompliant
    return random.random()


class Jar:
    def set_cookie(self, key, value):
        pass


def unrelated_receiver(response: Jar):
    response.set_cookie('session_id', str(random.getrandbits(128)))
