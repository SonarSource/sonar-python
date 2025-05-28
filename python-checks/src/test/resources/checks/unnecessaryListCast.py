
[i for i in list([1,2,3])] # Noncompliant {{Remove this unnecessary `list()` call on an already iterable object.}}
#           ^^^^

{i for i in list([1,2,3])} # Noncompliant
#           ^^^^

for i in list(range(2)): # Noncompliant 
#        ^^^^
    print(i)


some_iter = [1,2,5]
for i in list(some_iter): # Noncompliant 
    #    ^^^^
    print(i)

def foo(a: Union[str, Iterable[str]]):
  for i in list(a): # Noncompliant
      print(i)


list(range(32)) # Ok we only raise in for loops and comprehensions


for i in list(*some_iter): # Ok
    print(i)

from module import some_val

for i in list(some_val): # Noncompliant
    print(i)

#=============== COVERAGE =============

for i,y in list(range(3)), list(range(4)): 
    print(i)

for i in list([3,5], [1,3]): # incorrect syntax
    print(i)

for i in [1,2]:
    print(i)

from module import test

for i in test("1"):
    print(i)
