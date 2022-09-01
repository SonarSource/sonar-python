def for_loops():
    for _ in range(10):
        do_something()
    for _i in range(10): # Noncompliant
        do_something()
    for _myVaR in range(10):  # Noncompliant
        do_something()
    for dummy in range(10):  # Noncompliant
        do_something()
    for myignore in range(10):
        do_something()

