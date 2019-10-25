expression = 3 # Noncompliant@-1 {{File has a complexity of 4 which is greater than 2 authorized.}} [[effortToFix=2]]

def fun():
  if expression:
    pass
  if expression:
    pass
  if expression:
    pass
  return
