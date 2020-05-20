from math import *  # Noncompliant {{Import only needed names or import the module and then use its members.}}
import math
from math import exp
import math as m
from math import exp as e

if x:
    from os import * # Noncompliant
   #^^^^^^^^^^^^^^^^
    x = 10
else:
    from os import path

class Foo:
    pass

def foo():
    pass

x = 10
while x != 0:
    from os import * # Noncompliant

variable = []
variable[0] = 1
print(variable)
no_such_function(variable)
