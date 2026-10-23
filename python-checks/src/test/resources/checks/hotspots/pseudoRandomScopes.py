import random

generate_token = lambda: random.random()  # Noncompliant
verify = lambda x: x
verify_fn: "Callable" = lambda x: x
secure_hint: "Callable" = lambda x: x
id_value = random.random()  # names bound to a nested lambda belong to the lambda
shuffled = list(map(lambda _: random.random(), range(16)))

def build():
    token = list(map(lambda _: random.random(), range(16)))  # Noncompliant
    secret: list = list(map(lambda _: random.random(), range(16)))  # Noncompliant
    return token, secret

def aes256_key():
    random.random()  # Noncompliant

def pbkdf2_x():
    random.random()  # Noncompliant

def f(x):
    verify_input = lambda y: y
    plain = [lambda: random.random()]
    return random.random()

def derive_private_eccentric():
    random.random()
