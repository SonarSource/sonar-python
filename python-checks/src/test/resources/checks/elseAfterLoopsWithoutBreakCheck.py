def expectedIssues():
  for i in range(10):
      if i == 42:
          print('Magic number in range')
  else:   # Noncompliant {{Add a "break" statement or remove this "else" clause.}}
# ^^^^
      print('Magic number not found')


  for i in range(10):
      if i == 42:
          print('Magic number in range')
      else:
          raise ValueError("Foo")
  else:   # Noncompliant {{Add a "break" statement or remove this "else" clause.}}
# ^^^^
      print('Magic number not found')


  for i in range(10):
      if i == 42:
          print('Magic number in range')
          return i
  else:   # Noncompliant {{Add a "break" statement or remove this "else" clause.}}
# ^^^^
      print('Magic number not found')

  
  for i in range(10):
      if i == 42:
          print('Magic number in range')
  print('Magic number not found')

def sameForWhile():
  i = 0
  while i < 10:
      i = i + 1
      if i == 42:
          print('Magic number in range')
  else:   # Noncompliant {{Add a "break" statement or remove this "else" clause.}}
# ^^^^
      print('Magic number not found')

  i = 0
  while i < 10:
      i = i + 1
      if i == 42:
          print('Magic number in range')
      else:
          raise ValueError("Foo")
  else:   # Noncompliant {{Add a "break" statement or remove this "else" clause.}}
# ^^^^
      print('Magic number not found')


  i = 0
  while i < 10:
      i = i + 1
      if i == 42:
          print('Magic number in range')
          return i
  else:   # Noncompliant {{Add a "break" statement or remove this "else" clause.}}
# ^^^^
      print('Magic number not found')

  i = 0
  while i < 10:
      i = i + 1
      if i == 42:
          print('Magic number in range')
  print('Magic number not found')

def nestedLoops():
    for i in range(10):
      for j in range(10):
        if i == j:
          break # It's a break, but it breaks out of the wrong loop
    else: # Noncompliant {{Add a "break" statement or remove this "else" clause.}}
  # ^^^^
      print("hm...")

    for i in range(10):
      j = 0
      while j < 10:
        j = j + 1
        if i == j:
          break # It's a break, but it breaks out of the wrong loop
    else: # Noncompliant {{Add a "break" statement or remove this "else" clause.}}
  # ^^^^
      print("hm...")

def compliant():
  for i in range(10):
    if i > 5:
      break
  else:
    pass

# insprired by an actual false positive in twisted-12.1.0/twisted/internet/test/test_unix.py, line 450
def twistedNestedCompliant():
  for logEvent in []:
    for k, v in {'k': 'v'}:
      if v != k:
        break
    else:
        break
  else:
    pass
