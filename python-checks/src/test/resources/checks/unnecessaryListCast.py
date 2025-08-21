
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

def compliant_modifying_list_in_loop():
    some_list = [1, 2, 3]
    # Compliant, copying is necessary since the list is modified in the loop
    for i in list(some_list): 
        if i % 2 == 0:
            some_list.remove(i) 

    for i in list(some_list): some_list.append(5) 
    for i in list(some_list): some_list.extend([1, 2]) 
    for i in list(some_list): some_list.insert(0, 1) 
    for i in list(some_list): some_list.remove(i) 
    for i in list(some_list): some_list.pop() 
    for i in list(some_list): some_list.clear() 
    for i in list(some_list): some_list.sort() 
    for i in list(some_list): some_list.reverse() 
    for i in list(some_list): some_list.copy() # Noncompliant

def noncompliant_modifying_list_outside_loop():
    some_list = [1, 2, 3]
    for i in list(some_list): # Noncompliant
        print(i)
    some_list.append(5)

    [some_list.pop() for i in list(some_list)] # Noncompliant

    some_other_list = [10, 20, 30]
    for i in list(some_other_list): # Noncompliant
        some_list.append(i)

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

for i in list(test): # Noncompliant
    test().append(i)
    object.__setattr__(test, k, v)

for i in list(object): # Noncompliant
    test().append(i)
    object.__setattr__(test, k, v)

