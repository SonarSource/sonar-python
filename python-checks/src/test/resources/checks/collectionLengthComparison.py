def comparison_of_len_and_zero(a):
    if len(a) > 0: pass
    if len(a) < 0: pass # Noncompliant
    if len(a) >= 0: pass # Noncompliant
    if len(a) <= 0: pass # ok, this is a questionable construct, but the result is not known in advance.
    if len(a) == 0: pass

def reversed_comparison_of_len_and_zero(a):
    if 0 < len(a): pass
    if 0 > len(a): pass # Noncompliant
    if 0 <= len(a): pass # Noncompliant
    if 0 >= len(a): pass # ok, this is a questionable construct, but the result is not known in advance.
    if 0 == len(a): pass

def with_parentheses(a):
    if (len(a)) > 0: pass
    if (len(a)) >= 0: pass # Noncompliant
    if len(a) >= (0): pass # Noncompliant

def other_operators(a):
    if len(a) <> 0: pass
    if len(a) != 0: pass

def other_operands(a):
    if len(a) >= 2: pass
    if len(a) >= a: pass
    if 2 > len(a): pass
    if sum(a) >= 0: pass
    if xxx(a) >= 2: pass
    if a >= 0: pass
    if 0 >= 0: pass

def other_len(a, foo):
    len = foo
    if len(a) >= 0: pass
