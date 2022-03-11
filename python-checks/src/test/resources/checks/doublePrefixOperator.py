a = 1
b = not not a  # Noncompliant {{Use the ("not") operator just once or not at all.?}}
#   ^^^^^^^^^

c = ~~a  # Noncompliant {{Use the ("~") operator just once or not at all.?}}
#   ^^^
d = not (not (not (not a)))  # Noncompliant 3

e = ~~~~~a  # Noncompliant 4
f = ~(((((~a)))))  # Noncompliant

g = not (a == b)
h = ~(a and not b)
i = not a and ~b

j = not (not (a is not b))  # Noncompliant {{Use the ("not") operator just once or not at all.?}}
#   ^^^^^^^^^^^^^^^^^^^^^

k = not (~a)
