x=0
++x # Noncompliant {{This statement doesn't produce the expected result, replace use of non-existent pre-increment operator}}
print(x)

def foo(x):
    --x # Noncompliant {{This statement doesn't produce the expected result, replace use of non-existent pre-decrement operator}}
#   ^^^

print(x)
x+=1 # OK
print(x)
x-=1 # OK
print(x)
