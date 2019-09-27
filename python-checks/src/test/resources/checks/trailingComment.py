# More than one word
# Noncompliant@+1 {{Move this trailing comment on the previous empty line.}}
print("a") # More than one word
#          ^^^^^^^^^^^^^^^^^^^^
print("a") # OneWord
print("a")

def func(self):
# Noncompliant@+1
    if a==b: #Some comment
# Noncompliant@+1
        return None #This should not happen
    self.bar()

SOMEVAR = [
# Noncompliant@+1
  'asd', 'asd', # comment more than one word
  'asdpj'
]
