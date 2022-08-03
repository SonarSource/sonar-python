def foo(a, b, c):
    assert a, "ok"

    assert (a, ), "ok"
    assert (a, b), "ok"
    assert (a, b, c), "ok"
    assert (a, ) # Noncompliant {{Fix this assertion on a tuple literal.}}
#          ^^^^^
    assert () # Noncompliant

    assert (a, b) # Noncompliant {{Fix this assertion on a tuple literal.}}
#          ^^^^^^
    assert(a, b) # Noncompliant

    x = (a, b)
    assert x
