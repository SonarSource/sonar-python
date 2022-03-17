def start():
    a = 1
    b = not not a  # Noncompliant {{Use the "bool()" builtin function instead of calling "not" twice.}}
    #   ^^^^^^^^^

    c = ~~a  # Noncompliant {{Use the ("~") operator just once or not at all.?}}
    #   ^^^
    d = not (not (not (not a)))  # Noncompliant 3

    e = ~~~~~a  # Noncompliant
    f = ~(((((~a)))))  # Noncompliant {{Use the ("~") operator just once or not at all.?}}

    g = not (a == b)
    h = ~(a and not b)
    i = not a and ~b

    j = not (not (a is not b))  # Noncompliant {{Use the ("not") operator just once or not at all.?}}
    #   ^^^^^^^^^^^^^^^^^^^^^^

    k = not (~a)

    i = not(not(a)) # Noncompliant {{Use the ("not") operator just once or not at all.?}}

    class Foo:
        def __invert__(self):
            return self.bar == 42

    foo  = Foo()
    ~~foo # here ~~ it might have a different semantic
