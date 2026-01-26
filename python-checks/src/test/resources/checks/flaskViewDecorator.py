from flask.views import View

def login_required(f):
    return f

def cache(minutes):
    def decorator(f):
        return f
    return decorator

@login_required  # Noncompliant {{Move this decorator to the "decorators" class attribute.}}
#^[sc=1;ec=15]
class UserList(View):
    def dispatch_request(self):
        return "users"

@login_required  # Noncompliant
@cache(minutes=2)  # Noncompliant
class CachedUserList(View):
    def dispatch_request(self):
        return "cached users"

class CompliantUserList(View):
    decorators = [login_required, cache(minutes=2)]

    def dispatch_request(self):
        return "users"

class NoDecoratorView(View):
    def dispatch_request(self):
        return "no decorators"

@login_required
class RegularClass:
    pass

class CustomView(View):
    pass

@login_required  # Noncompliant
class DerivedView(CustomView):
    def dispatch_request(self):
        return "derived"

class CompliantDerivedView(CustomView):
    decorators = [login_required]

    def dispatch_request(self):
        return "compliant derived"
