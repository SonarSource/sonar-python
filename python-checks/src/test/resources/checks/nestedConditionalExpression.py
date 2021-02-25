def f(a, b, c):
  print(a if b else c)
  print(a if b else c if b else c) # Noncompliant
  print((a if b else c) if b else c) # Noncompliant
#        ^^^^^^^^^^^^^  ^^< {{Parent conditional expression.}}
  print(a if (a if b else c) else c) # Noncompliant
#             ^^^^^^^^^^^^^
  print(a if b else (a if b else c)) # Noncompliant
#                    ^^^^^^^^^^^^^
  print(a if b else 42 + (a if b else c)) # Noncompliant
#                         ^^^^^^^^^^^^^
  print((a if b else (a if b else c)) if b else c) # Noncompliant 2

  print([a if b else c for a in range(5)] if b else c)
  print({a if b else c for a in range(5)} if b else c)
  print((a if b else c for a in range(5)) if b else c)
  print({'key': a if b else c for a in range(5)} if b else c)

  print([(a if b else c) if b else c for a in range(5)])
