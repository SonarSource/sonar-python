from a_module import tuple_from_somewhere, create_tuple

def test_tuple():
    t = tuple() # Compliant
    t = tuple(*[[1,2,3]]) # FP but rare enough to not cover it
    t = tuple(tuple_from_somewhere) # Compliant
    t = tuple(create_tuple()) # Compliant

    t1 = tuple((1, 2)) # Noncompliant {{Remove the redundant tuple constructor call.}}
    #    ^^^^^

    t2 = tuple((i for i in (1, 2))) # Compliant (i for i in (1, 2)) is a generator expression, not a tuple

    t3 = tuple([1,2,3]) # Noncompliant {{Replace this tuple constructor call by a tuple literal.}}

    t4 = tuple({1,2,3}) # Noncompliant {{Replace this tuple constructor call by a tuple literal.}}

    t5 = tuple({"a":1, "b":2}) # Noncompliant {{Replace this tuple constructor call by a tuple literal.}}

    t_not_transforming = tuple([x for x in [1,2]]) # Compliant, comprehension is not transforming data

    t6 = tuple([2*x for x in [1,2]]) # Noncompliant {{Replace this list comprehension by a generator.}}
    #          ^^^^^^^^^^^^^^^^^^^^

    t7 = tuple({x**2 for x in [1,2]}) # Noncompliant {{Replace this set comprehension by a generator.}}
    #          ^^^^^^^^^^^^^^^^^^^^^

    t8 = tuple({x:x for x in [1,2]}) # Noncompliant {{Replace this dict comprehension by a generator.}}
    #          ^^^^^^^^^^^^^^^^^^^^

def test_list():
    l = list(x * 2 for x in [1,2,3]) # Compliant

    l1 = list([x * 2 for x in [1,2,3]]) # Noncompliant {{Remove the redundant list constructor call.}}
    #    ^^^^

    l2 = list([1,2,3]) # Noncompliant {{Remove the redundant list constructor call.}}

    l3 = list((1,2,3)) # Noncompliant {{Replace this list constructor call by a list literal.}}

    l4 = list({1,2,3}) # Noncompliant {{Replace this list constructor call by a list literal.}}

    l5 = list({"a":1, "b":2}) # Noncompliant {{Replace this list constructor call by a list literal.}}

    l_not_transforming = list({x for x in [1,2]}) # Compliant, comprehension is not transforming data

    l6 = list({x%2 for x in [1,2,3,4]}) # Compliant: It can makes sense to use first a set comprehension then a list to avoid duplicate for example

    l7 = list({x:x for x in [1,2]}) # Noncompliant {{Replace this list constructor call by a list literal.}}

def test_set():
    s = set(x * 2 for x in [1,2,3]) # Compliant

    s1 = set({x * 2 for x in [1,2,3]}) # Noncompliant {{Remove the redundant set constructor call.}}
    #    ^^^

    s2 = set({1,2,3}) # Noncompliant {{Remove the redundant set constructor call.}}

    s3 = set((1,2,3)) # Noncompliant {{Replace this set constructor call by a set literal.}}

    s4 = set([1,2,3]) # Noncompliant {{Replace this set constructor call by a set literal.}}

    s5 = set({"a":1, "b":2}) # Noncompliant {{Replace this set constructor call by a set literal.}}

    s_not_transforming = set([x for x in [1,2]]) # Compliant, not transforming data

    s6 = set([x for x in [1,2] if x%2==0]) # Noncompliant {{Replace this set constructor call by a set literal.}}

    s7 = set({x:x for x in [1,2]}) # Noncompliant {{Replace this set constructor call by a set literal.}}

def test_dict():
    d1= dict({x : x for x in [1,2,3]}) # Noncompliant {{Remove the redundant dict constructor call.}}
    #   ^^^^

    d2 = dict({1:1, 2:2, 3:3}) # Noncompliant {{Remove the redundant dict constructor call.}}

    d3 = dict([(1,2), (3,4)]) # Noncompliant {{Replace this dict constructor call by a dict literal.}}

    d4 = dict({(1,2), (3,4)}) # Noncompliant {{Replace this dict constructor call by a dict literal.}}

    d5 = dict(((1,2), (3,4))) # Noncompliant {{Replace this dict constructor call by a dict literal.}}

    d6 = dict([(x,x**2) for x in [1,2]]) # Noncompliant {{Replace this dict constructor call by a dict literal.}}

    d7 = dict({(x,x**2) for x in [1,2,3,1]}) # Compliant, could make sense to remove duplicates before creating a dict