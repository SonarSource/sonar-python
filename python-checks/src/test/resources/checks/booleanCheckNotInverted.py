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

p = not (a==1) == b == c == (d and e)

q = not (a and 1) == b == c == (d and e)
