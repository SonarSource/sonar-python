if a:
  if b:
    pass
  if c:
    pass

if a:
  if b:
    pass
  else:
    pass

if a:
  if b: # Noncompliant {{Merge this if statement with the enclosing one.}}
# ^^
    pass

if a:
  pass
elif b:
  if c: # Noncompliant
    pass

if a:
  pass
elif b:
  if c:
    pass
else:
  pass

# Noncompliant@+2
if a:
  if b:
    if c: # Noncompliant
      pass

if a:
  if b: # Noncompliant
    if c:
      pass
    else:
      pass

if a:
  if b:
    pass
  elif c:
    pass

if a:
  if b:
    pass
  do_something()


if is_first_condition and is_second_condition:
  if is_third_condition and is_fourth_condition:  # merging two ifs in a single line would break line length
    ...

if cond:
  if cond2 := foo():
    ...

if cond := foo():
  if cond2:
    ...

if cond1:
  # comment specific to cond2
  if cond2:
    ...
