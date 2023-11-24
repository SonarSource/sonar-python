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

var = (some.method(arg1, arg2)
           # Noncompliant@+1
        if hasattr(umath, 'nextafter')  # Missing on some platforms?
        else float64_ma.huge)

#Noncompliant@+1
with errstate(over='ignore'): #some comment
    if bar:
        print("hello")

# No issue for common pragma comments
my_dict = {
    'a': 1, 'b': 2,
    'c': 3, 'z': 0
}  # fmt: skip

import frobnicate  # type: ignore
frobnicate.start()

# Flake8 pragma comments, such as the following are ok.
example = lambda: 'example' # noqa: E731
