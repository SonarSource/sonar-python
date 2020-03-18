class A:
  _foo = 42 # Noncompliant {{Remove this unread private attribute '_foo' or refactor the code to use its value.}}
# ^^^^
