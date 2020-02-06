from argumentNumberImported import fn

fn(1, 2) # Noncompliant
fn(1) # OK

def stub_functions():
    from flask import make_response
    make_response(1, 2) # OK, no issue raised on external functions
