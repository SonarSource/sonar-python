from argumentNumberImported import fn, A
import argumentNumberImported

fn(1, 2) # Noncompliant
fn(1) # OK

def stub_functions():
    from flask import make_response
    make_response(1, 2) # OK, no issue raised on external functions
    a = A()
    a.meth() # Noncompliant
    a.meth(42)
    a.meth(42, 43) # Noncompliant
    argumentNumberImported.fn(1)
