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

changes = sa.Table('foo', bar,
    # Noncompliant@+1
    sa.Column('id', sa.Integer), # yet another comment
    # some comment
    sa.Column('name', sa.String(256)),
    )

#Noncompliant@+1
toto = (state, None, # a comment
        [foo(b) for b in bar])
foo.bar(
         SomeClass(workdir='wkdir',
                      command=['cmd',
                               # Noncompliant@+1
                               'foo'], # note extra param
                      env=some.method(
                          r'sf',
                          l='l', p='p', i='i'))
          + 0
      )
