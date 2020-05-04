import jinja2
from jinja2 import Environment

myEnvironment = foo()
env = myEnvironment()

env = Environment() # Noncompliant {{Remove this configuration disabling autoescape globally.}}
env = Environment(True) # Noncompliant
env = Environment(autoescape=True)
env = Environment(xx=False,autoescape=True)
env = Environment(autoescape=False) # Noncompliant
env = jinja2.Environment(autoescape=False)  # Noncompliant
env = Environment(autoescape=select_autoescape(['html']))

env = Environment(**xxx)

options1 = {'autoescape': True}
env = Environment(**options1)

options2 = {'autoescape': False}
env = Environment(**options2)  # Noncompliant

options3 = {'myoption': True}
env = Environment(**options3)  # Noncompliant

options4 = {'autoescape': foo()}
env = Environment(**options4)

options5 = {'autoescape': True}
options5 = {'autoescape': False}
env = Environment(**options5) # FN

options6 = foo()
env = Environment(**options6)

options7 = {'my' + 'option': True}
env = Environment(**options7) # Noncompliant

options8 = {**xxx}
env = Environment(**options8) # Noncompliant

TEMPLATES = [
    {
        'BACKEND': 'xxx',
        'OPTIONS': { 'autoescape': False } # OK: file name is not settings.py
    }
]

