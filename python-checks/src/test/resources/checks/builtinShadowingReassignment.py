max = 42

def abc():
    max = 42  # Noncompliant {{Rename this variable; it shadows a builtin.}}
   #^^^
    max = foo()
   #^^^<

def bcd():
    max = 42  # Noncompliant {{Rename this variable; it shadows a builtin.}}
   #^^^
