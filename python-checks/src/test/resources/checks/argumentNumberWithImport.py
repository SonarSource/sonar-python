from argumentNumberImported import fn, A
import argumentNumberImported
from argumentNumberImported import reassigned_unbound_meth as aliased_unbound
from argumentNumberImported import reassigned_bound_meth as aliased_bound

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


def calling_reassigned_imported_valid_usages():
    # SONARPY-2285: Bound and unbound methods are currently not exported
    argumentNumberImported.reassigned_unbound_meth(1, 2)
    argumentNumberImported.reassigned_bound_meth(1)

    aliased_unbound(1, 2)
    aliased_bound(1)

def calling_reassigned_imported_noncompliant():
    # SONARPY-2285: Bound and unbound methods are currently not exported
    argumentNumberImported.reassigned_unbound_meth() # FN SONARPY-2285
    argumentNumberImported.reassigned_unbound_meth(42, 43) # FN SONARPY-2285
    argumentNumberImported.reassigned_bound_meth() # FN SONARPY-2285
    argumentNumberImported.reassigned_bound_meth(42, 43) # FN SONARPY-2285

    aliased_unbound() # FN SONARPY-2285
    aliased_unbound(1, 2, 3) # FN SONARPY-2285

    aliased_bound() # FN SONARPY-2285
    aliased_bound(42, 43) # FN SONARPY-2285

