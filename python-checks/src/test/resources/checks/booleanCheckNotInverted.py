a = 1
b = 2
c = not (a == b)  # Noncompliant {{Use the opposite operator ("!=") instead.}}
#   ^^^^^^^^^^^^
cc = not (a != b)  # Noncompliant

d = not (a > b)  # Noncompliant

e = not (a <= b)  # Noncompliant

f = float('nan')

g = not (f > 2)  # Noncompliant

h = not (f >= 2)  # Noncompliant

i = not (((((a * 2)))))

j = a and (not b)

k = (not a) and (not b)

l = not ((((((a > b))))))  # Noncompliant

o = not a == b  # Noncompliant

m = not a == b == 1

n = not (not (a == b))  # Noncompliant {{Use the opposite operator ("!=") instead.}}
#        ^^^^^^^^^^^^
p = not (1 == a == b) == 2 # Noncompliant {{Use the opposite operator ("!=") instead.}}
#   ^^^^^^^^^^^^^^^^^^^^^^
q = not (a == 1) == b == c == (d and e)

r = not (a and 1) == b == c == (d and e)
s = a is not b

t = not (a is b)  # Noncompliant {{Use the opposite operator ("is not") instead.}}
#   ^^^^^^^^^^^^
u = not (a is not b)  # Noncompliant {{Use the opposite operator ("is") instead.}}
#   ^^^^^^^^^^^^^^^^
v = not (a is (not b))  # Noncompliant {{Use the opposite operator ("is not") instead.}}
#   ^^^^^^^^^^^^^^^^^^
x = a is not (not b)

list_ = [0, 2, 3]
y = not (1 in list_)  # Noncompliant
z = not (1 not in list_)  # Noncompliant

# Both cases below are handled by RSPEC-2761
## x = not(not 1) # Noncompliant
## t = a is not(not b) # Noncompliant

def func():
    if not a == 2:     #Noncompliant
#      ^^^^^^^^^^
        b = 10
    return "item1" "item2"

def func1():
    if not a == 2 and b == 9:   # Noncompliant
#      ^^^^^^^^^^
        b = 10
    return "item1" "item2"

def func2():
    if a != 2 and not b == 9:   # Noncompliant
#                 ^^^^^^^^^^
        b = 10
    return "item1" "item2"

def set_comparison_test():
    from typing import Set
    def validate(a: Set[str], b: Set[str]) -> None:
        if not (a <= b): # OK
            ...

        if not (a == b): # Noncompliant
            ...

    ingredients: Set[str] = {'beef', 'potatos'}
    consumable_by_vegans: Set[str] = {'nut', 'apple', 'potatos'}
    # The example below is a FP. The type of both sets is inferred as ANY. Should be fixed as part of:
    # https://sonarsource.atlassian.net/browse/SONARPY-1574
    if not (ingredients <= consumable_by_vegans): # Noncompliant
        ...

    ingredients_not_annotated = {'beef', 'potato'}
    consumable_by_vegans_not_annotated = {'nut', 'apple', 'potato'}
    if not (ingredients_not_annotated < consumable_by_vegans_not_annotated):       # OK
        ...

    validate(ingredients, consumable_by_vegans)
