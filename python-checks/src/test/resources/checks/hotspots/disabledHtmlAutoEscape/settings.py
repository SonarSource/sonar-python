# Django template engine classes are loaded dynamically
# Values are loaded from the global TEMPLATES list
def foo():
    return 42

TEMPLATES = [
    {
        'BACKEND': 'xxx',
        'OPTIONS': { 'autoescape': False }, # Noncompliant {{Make sure disabling auto-escaping feature is safe here.}}
#                    ^^^^^^^^^^^^^^^^^^^
    },
]
TEMPLATES = [
    {
        'BACKEND': 'xxx',
        'OPTIONS': { 'autoescape': True }
    }
]
TEMPLATES = [
    {
        'BACKEND': 'xxx',
        'OPTIONS': { 'abc': False }
    }
]
TEMPLATES = [
    {
        'BACKEND': 'xxx',
        'OPTIONS': { 'autoescape': foo() }
    }
]

# Preserve dictionary-display-only handling when comprehensions also expose key/value pair nodes.
options = {'autoescape': False for template in templates}
options = {template: {'autoescape': False} for template in templates}  # Noncompliant
