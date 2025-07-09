
def func(x):
    return x * 2

def lists():
    list(func(x) for x in range(5))  # Noncompliant {{Replace list constructor call with a list comprehension.}}
   #^^^^

    list(x + 1 for x in range(10))  # Noncompliant
    list((x, y) for x in range(3) for y in range(2))  # Noncompliant
    list(x for x in range(10) if x % 2 == 0)  # Noncompliant

    list(x for x in range(10))
    list(range(10))
    list()
    list([1, 2, 3])

def sets():
    set(func(x) for x in range(5))  # Noncompliant {{Replace set constructor call with a set comprehension.}}
   #^^^
    set(x * 2 for x in items)  # Noncompliant
    set(x for x in range(10) if x > 5)  # Noncompliant

    set(x for x in range(10))
    set()
    set([1, 2, 3])
    set(x for x,y in range(10)) # Noncompliant
    set(y for x in range(10)) # Noncompliant

def create_tuple(t) -> tuple: ...
def create_non_tuple(t) -> str: ...

def dicts():
    dict((k, v) for k, v in items.items())
    dict((k, func(v)) for k, v in items.items())  # Noncompliant {{Replace dict constructor call with a dictionary comprehension.}}
   #^^^^
    dict((v, k) for k, v in items.items())  # Noncompliant
    dict((k.upper(), v) for k, v in items.items())  # Noncompliant
    dict((k, v) for k, v in items.items() if k.startswith('a'))  # Noncompliant
    dict((t, t) for t in items) # Noncompliant
    dict((t, t, t) for t in items) # Noncompliant
    dict((t,) for t in items) # Noncompliant

    dict(tuple(k, v) for k, v in items.items() if k.startswith('a'))  # OK
    dict(tuple([k, v]) for k, v in items.items()) # OK
    dict(create_tuple(t) for t in items) # OK
    dict(create_non_tuple(t) for t in items) # Noncompliant
    # FP SONARPY-3154
    dict(x.split("=") for x in update_str.split(",")) # Noncompliant

def compliant_examples():
    [func(x) for x in range(5)]
    {func(x) for x in range(5)}
    {k: func(v) for k, v in items.items()}
    {v: k for k, v in items.items()}

def other():
    list(map(func, range(10)))
    list([x for x in range(10)])
    result = list(func(x) for x in range(5))  # Noncompliant

    list(*args)
    list(**args)

def multiple_clauses():
    list((i, j) for i in range(3) for j in range(2))  # Noncompliant
    [(i, j) for i in range(3) for j in range(2)]
