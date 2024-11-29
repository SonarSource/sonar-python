
def param_with_type_hint(a: int):
    a() # OK

def call_unknown_value():
    (~s)(X) # Ok
