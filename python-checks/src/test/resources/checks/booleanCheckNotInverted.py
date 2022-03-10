a = 1
b = 2
c = not (a == b)  # Noncompliant {{Use the opposite operator ("!=") instead.}}
#   ^^^^^^^^^^^^

d = not (a > b)  # Noncompliant

e = not (a <= b)  # Noncompliant

f = float('nan')

g = not (f > 2)  # Noncompliant

i = not (((((a * 2)))))

j = a and (not b)

k = (not a) and (not b)

l = not ((((((a > b))))))  # Noncompliant

n = not (not (a == b))  # Noncompliant

o = not a == b  # Noncompliant

m = not a == b == 1

p = not (a == 1) == b == c == (d and e)

q = not (a and 1) == b == c == (d and e)
r = a is not b

r = not (a is b)  # Noncompliant {{Use the opposite operator ("is not") instead.}}
s = not(a is not b)  # Noncompliant
t = not(a is (not b)) # Noncompliant
t = a is not(not b)

list_ = [0, 2, 3]
u = not (1 in list_)  # Noncompliant
v = not (1 not in list_)  # Noncompliant

# Both below should be handled by ticket SONARPY-253
## x = not(not 1) # Noncompliant
## t = a is not(not b) # Noncompliant
