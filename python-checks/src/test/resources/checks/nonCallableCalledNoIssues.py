
def param_with_type_hint(a: int):
    a() # OK

def call_unknown_value(unknown_value):
    (~unknown_value)(X) # Ok
