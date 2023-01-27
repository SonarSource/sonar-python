a = (1 if x else 1) if cond else (1 if x else 1) # Noncompliant

def func():
  if b == 0:  # Noncompliant {{Remove this if statement or edit its code blocks so that they're not all the same.}}
# ^^
    doOneMoreThing()
#   ^^^^^^^^^^^^^^^^<
  elif b == 1:
    doOneMoreThing()
#   ^^^^^^^^^^^^^^^^<
  else:
    doOneMoreThing()
#   ^^^^^^^^^^^^^^^^<

def func():
  if b == 0:  # Noncompliant {{Remove this if statement or edit its code blocks so that they're not all the same.}}
# ^^
    doSomething()
#   ^[el=+2;ec=20]<
    doOneMoreThing()
  else:
    doSomething()
#   ^[el=+2;ec=20]<
    doOneMoreThing()

if b == 0:  # ok, exception when no else clause
  doOneMoreThing()
elif b == 1:
  doOneMoreThing()

if b == 0:  # ok, not all branches are the same
  doSomething()
elif b == 1:
  doSomethingElse()
else:
  doSomething()

if b == 0:  # ok, not all branches are the same
  doSomething()
elif b == 1:
  doSomething()
else:
  doSomethingElse()


if b == 0:  # ok
  doSomething()
elif b == 1:
  doSomethingElse()


a = 1 if x else 1 # Noncompliant
#   ^>^^        ^<

a = (lambda x: x+1
#              ^^^>
     if x > 0 # Noncompliant
#    ^^
     else x+1)
#         ^^^<

a = 1 if x else 1 if y else 1 # Noncompliant

a = 1 if x else 1 if y else 1 if z else 1 # Noncompliant

a = (1 if x else 1) if cond else (2 if x else 3) # Noncompliant
#    ^>^^        ^<
a = 1 if x else 2 if y else 1

a = 1 if x else 1 if y else 2

a = (1 if x else 1) if cond else 1 # Noncompliant

a = (1 if x else 1) if cond else (1 if x else 1) # Noncompliant

a = ((1 if x else 1) if cond else (1 if x else 1)) if other else (1 if x else (1 if y else 1 if z else 1)) # Noncompliant
#     ^>          ^>               ^>          ^>  ^^             ^<           ^<          ^<          ^<

a = 1 if (x and y) else 1 # Noncompliant

def test_secondary_messages():
    if b == 0: # Noncompliant
#   ^^
        doSomething()
#       ^^^^^^^^^^^^^< 1 {{Duplicated statements.}}
    else:
        doSomething()
#       ^^^^^^^^^^^^^< 2 {{Duplicated statements.}}
