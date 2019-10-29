def f(a, b, c):
  print(a if b else c)
  print(a if b else c if b else c) # Noncompliant
  print((a if b else c) if b else c) # Noncompliant
#        ^^^^^^^^^^^^^
  print(a if (a if b else c) else c) # Noncompliant
#             ^^^^^^^^^^^^^
  print(a if b else (a if b else c)) # Noncompliant
#                    ^^^^^^^^^^^^^
  print(a if b else 42 + (a if b else c)) # Noncompliant
#                         ^^^^^^^^^^^^^
  print((a if b else (a if b else c)) if b else c) # Noncompliant 2
