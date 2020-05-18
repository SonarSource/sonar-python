from math import *  # Noncompliant {{Import only needed names or import the module and then use its members.}}
import math
from math import exp

if x:
    from os import * # Noncompliant
   #^^^^^^^^^^^^^^^^
else:
    from os import path
