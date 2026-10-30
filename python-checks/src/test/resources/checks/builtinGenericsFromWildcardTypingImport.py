from typing import *

def foo(items: List[int]): # Noncompliant {{Use the built-in generic type `list` instead of its typing counterpart.}}
              #^^^^^^^^^
    pass
