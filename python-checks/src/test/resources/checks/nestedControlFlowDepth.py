for y in range(10):
    if x > y:
        if x > 1:
            pass

for x in range(10):
#^[sc=1;ec=3]>
    while x:
#   ^^^^^>
        for y in range(10):
#       ^^^>
            if x > y:
#           ^^>
                if x > 1:  # Noncompliant {{Refactor this code to not nest more than 4 "if", "for", "while", "try" and "with" statements.}} 
#               ^^
                    if y > 10:
                        pass

def fun():
  if cond1:
      for user in results:
          if user:
              x = "foo"
  elif cond2:
      for user in results:
          if user:
              for key in user:
                  if lastCond: # Noncompliant
                     pass
              x ="bar"
          else:
              x = "qix"
  return x

for y in range(10):
    if x > y:
        if x > 1:
            pass

for x in range(10):
    try:
        for y in range(10):
            if x > y:
              async with x.bar() as foo:   # Noncompliant
#                   ^^^^
                pass
    except Error:
        pass
