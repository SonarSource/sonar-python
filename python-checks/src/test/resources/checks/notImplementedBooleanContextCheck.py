def test_if_statement():
    result = NotImplemented
    if result:  # Noncompliant {{NotImplemented should not be used in boolean contexts.}}
    #  ^^^^^^
        pass

    if not result:  # Noncompliant
    #      ^^^^^^
        pass

    if result is True:  # Noncompliant
        pass

# Boolean operations
def test_boolean_operations():
    x = NotImplemented and True  # Noncompliant
    y = False or NotImplemented  # Noncompliant
    z = NotImplemented or NotImplemented  # Noncompliant 2

# Explicit boolean conversions
def test_bool_conversion():
    x = bool(NotImplemented)  # Noncompliant
    y = not NotImplemented  # Noncompliant

# Special methods returning NotImplemented
class TestClass:
    def __eq__(self, other):
        result = NotImplemented
        if result:  # Noncompliant
            return True
        return False

    def __lt__(self, other):
        result = NotImplemented
        return bool(result)  # Noncompliant

    def __gt__(self, other):
        result = NotImplemented
        if result is NotImplemented:  # Compliant
            return False
        return True

    def __le__(self, other):
        result = NotImplemented
        while result:  # Noncompliant
            pass

# Edge cases and complex scenarios
def test_edge_cases():
    result = NotImplemented

    # Complex boolean expressions
    x = True and (result or False)  # Noncompliant
    y = not (result and True)  # Noncompliant

    # Multiple conditions
    if result and True or False:  # Noncompliant
        pass

    # Ternary operations
    z = True if result else False  # Noncompliant

    # List comprehension with boolean context
    items = [x for x in [1, 2, 3] if result]  # Noncompliant

# Valid uses of NotImplemented
def test_valid_uses():
    result = NotImplemented
    
    # Compliant - Identity comparisons
    if result is NotImplemented:
        pass

    if result is not NotImplemented:
        pass

    # Compliant - Equality comparisons
    if result == NotImplemented:
        pass

    if result != NotImplemented:
        pass

    return result

def coverage():
    def foo():
        return NotImplemented

    if foo() is True: 
        pass

    if False is foo(): 
        pass

    if foo() is foo(): 
        pass

# Public identity must also work through imports and qualified access.
def public_identity(flag, unknown):
    import builtins
    from builtins import NotImplemented as sentinel, bool as boolean
    if builtins.NotImplemented:  # Noncompliant
        pass
    boolean(sentinel)  # Noncompliant
    alias = sentinel
    if alias:  # Noncompliant
        pass
    mixed = sentinel if flag else False
    if mixed:  # Unknown alternative: do not report.
        pass
    if unknown:
        pass

def shadowed_builtin(bool, NotImplemented):
    bool(NotImplemented)
    if NotImplemented:
        pass

# Boolean operands are reported individually, without another issue on the compound expression.
def compound_conditions():
    import builtins
    from builtins import NotImplemented as sentinel

    if NotImplemented or NotImplemented:  # Noncompliant 2
        pass
    if (NotImplemented and NotImplemented):  # Noncompliant 2
        pass
    if ((
        builtins.NotImplemented  # Noncompliant
       #^^^^^^^^^^^^^^^^^^^^^^^
        or sentinel  # Noncompliant
        #  ^^^^^^^^
    )):
        pass
    if NotImplemented or (NotImplemented and NotImplemented):  # Noncompliant 3
        pass
    while NotImplemented and NotImplemented:  # Noncompliant 2
        break
    bool(NotImplemented or NotImplemented)  # Noncompliant 2
    not (NotImplemented and NotImplemented)  # Noncompliant 2
    [x for x in [] if NotImplemented or NotImplemented]  # Noncompliant 2
    1 if NotImplemented and NotImplemented else 0  # Noncompliant 2
    (NotImplemented or NotImplemented) is True  # Noncompliant 2

    # A conditional expression has no operand visitor: keep reporting its result.
    if (NotImplemented if True else NotImplemented):  # Noncompliant
        pass
