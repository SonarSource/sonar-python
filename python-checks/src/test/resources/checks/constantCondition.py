def literals():
  if 42: ... # Noncompliant {{Replace this expression; used as a condition it will always be constant.}}
#    ^^
  if False: ... # Noncompliant
  if 'a string': ... # Noncompliant
  if b'bytes':  ... # Noncompliant
  if {}:  ... # Noncompliant
  if {"a": 1, "b": 2}:  ... # Noncompliant
  if {41, 42, 43}: ... # Noncompliant
  if []:  ... # Noncompliant
  if [41, 42, 43]:  ... # Noncompliant
  if (41, 42, 43):  ... # Noncompliant
  if ():  ... # Noncompliant
  if None:  ... # Noncompliant

def unpacking(p1, p2):
  if ["a string", *p1, *p2]:  ... # Noncompliant
  if [*p1, *p2]: ... # OK, it may be empty or not
  if {"key": 1, **p1, **p2}:  ... # Noncompliant
  if {**p1, **p2}:  ... # OK, it may be empty or not
  if {"key", *p1, *p2}: ... # Noncompliant
  if {*p1, *p2}:  ... # OK, it may be empty or not
  if ("key", *p1, *p2):  ... # Noncompliant
  if (*p1, *p2):  ... # OK, it may be empty or not

def conditional_expr():
  var = 1 if 2 else 3  # Noncompliant

def boolean_expressions():
  if input() or 3:  ... # Noncompliant
#               ^
  if 3 and input():  ... # Noncompliant
  if 3 + input():  ... # OK
  if foo() and bar():  ... # OK
  if not 3:  ... # Noncompliant
  if not input(): ...


  3 + input()
  var = input() or 3  # Ok. 3 does not act as a condition when it is the last value of an "or" chain.
  var = input() and 3  # Ok. 3 does not act as a condition when it is the last value of an "and" chain.


  var = input() and 3 and input()  # Noncompliant
#                   ^
  var = input() or 3 or input()  # Noncompliant
  var = input() and 3 or input()  # Ok. 3 is the return value when the first input() is True.
  var = input() or 3 and input()  # Noncompliant
  var = 3 or input() and input()  # Noncompliant
  var = 3 and input() or input()  # Noncompliant

def ignored():
  # while loops are out of scope
  while True:
    pass
  # builtin constructors are out of scope
  if list():
    pass