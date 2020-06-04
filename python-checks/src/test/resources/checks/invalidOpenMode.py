from unknown import open as other_open

def compliant(path):
  with open(path, "a") as f: ...
  with other_open(path, "aw") as f: ... # OK
  open(path, "a")
  open(path, "r")
  open(path, "w")
  open(path, "x")
  open(path, "ab")
  open(path, "rb")
  open(path, "wb")
  open(path, "xb")
  open(path, "at")
  open(path, "rt")
  open(path, "wt")
  open(path, "xt")
  open(path, "ab+")
  open(path, "a+b")
  open(path, "+ab")
  open(path, "a+b")
  open(path, "+ta")
  open(path, "br")
  open(path, "xb")
  open(path, "x+b")
  open(path, "Ut")
  open(path, "Ubr")
  open(path) # OK, default to 'r'

def non_compliant(path):
  with open(path, "aw") as f: ... # Noncompliant {{Fix this invalid mode string.}}
  #               ^^^^
  open(path, "+rUb") # Noncompliant
  open(path, "wtb")  # Noncompliant
  open(path, "rwx")  # Noncompliant
  open(path, "ww")  # Noncompliant
  open(path, "+")  # Noncompliant
  open(path, "xw")  # Noncompliant
  open(path, "Uw") # Noncompliant
  open(path, "Ux") # Noncompliant
  open(path, "Ua")  # Noncompliant
  open(path, "Uxaw+")  # Noncompliant
  open(path, "Ur++")  # Noncompliant
  open(path, "z")  # Noncompliant
  open(path, "aaaaaaaaaaaaaaaaaaaa") # Noncompliant

def edge_cases(path):
  my_mode = "unrelated"
  #         ^^^^^^^^^^^>
  open(path, my_mode) # Noncompliant {{Fix this invalid mode string.}}
  #          ^^^^^^^
  open(path, f"{my_mode}") # FN
  open(path, get_mode())
  a_mode = get_mode()
  open(path, a_mode)
  multiple_assigned = "aw"
  multiple_assigned = "wtb"
  open(path, multiple_assigned) # FN

