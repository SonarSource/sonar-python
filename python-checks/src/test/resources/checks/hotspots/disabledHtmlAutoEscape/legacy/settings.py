# Django template engine classes are loaded dynamically
# Values are loaded from the global TEMPLATES list
def foo():
    return 42

TEMPLATES = [
    {
        'BACKEND': 'xxx',
        'OPTIONS': { 'autoescape': False }, # Noncompliant {{Remove this configuration disabling autoescape globally.}}
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
