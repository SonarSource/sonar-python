
def param_with_type_hint(a: int):
    a() # OK

def call_unknown_value(unknown_value):
    (~unknown_value)(X) # Ok

import socket
def type_alias_is_callable():
    # socket.error is a TypeAlias for OSError in typeshed — calling it must not trigger S5756
    e = socket.error(42, "connection refused")  # OK
