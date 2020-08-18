from typing import List

class A: ...

def custom(param: A):
  param[42] # Noncompliant {{Fix this "__getitem__" operation; Previous type checks suggest that "param" does not have this method.}}
  param[42] = 42 # Noncompliant {{Fix this "__setitem__" operation; Previous type checks suggest that "param" does not have this method.}}
  del param[42] # Noncompliant {{Fix this "__delitem__" operation; Previous type checks suggest that "param" does not have this method.}}

def builtin(param1: memoryview, param2: frozenset, param3: List[int]):
  del param1[0]  # Noncompliant
  param2[42] = 42 # Noncompliant
  param3[0]

def derived(param1: int, param2: int, *param3: int):
  (param1 + param2)[0] # Noncompliant {{Fix this "__getitem__" operation; Previous type checks suggest that this expression does not have this method.}}
  param3[42] # OK
