from typing import List
class A: ...

def custom(param1: A, param2: A):
  param1 + param2 # Noncompliant {{Fix this "+" operation; Previous type checks suggest that operands have incompatible types (A and A).}}
  #      ^
  -param1 # Noncompliant {{Fix this "-" operation; Previous type checks suggest that operand has incompatible type.}}

def builtin(param1: int, param2: str, param3: List[str]):
  param1 + param2 # Noncompliant {{Fix this "+" operation; Previous type checks suggest that operands have incompatible types (int and str).}}
  param1 + param3 # Noncompliant
