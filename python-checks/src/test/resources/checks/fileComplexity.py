expression = 3 # Noncompliant@-1 [[effortToFix=2]] {{File has a complexity of 4 which is greater than 2 authorized.}}

def fun():
  if expression:
    pass
  if expression:
    pass
  if expression:
    pass
  return
