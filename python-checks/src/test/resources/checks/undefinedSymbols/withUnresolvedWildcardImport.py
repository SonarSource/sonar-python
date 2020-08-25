from another_mod import *
from external_lib import *

def f():
    print(a) # OK
    print(c) # OK, it has unresolved wildcard import
    print(external_a) # OK