def f(*args):
    pass


def noncompliant(groups):
    f(*x for x in groups)  # Noncompliant {{Add parentheses around this starred generator argument.}}
    # ^^^^^^^^^^^^^^^^^^
    obj.m(*x for x in groups)  # Noncompliant {{Add parentheses around this starred generator argument.}}
    #     ^^^^^^^^^^^^^^^^^^


async def noncompliant_async(groups):
    f(*x async for x in groups)  # Noncompliant {{Add parentheses around this starred generator argument.}}
    # ^^^^^^^^^^^^^^^^^^^^^^^^


@f(*x for x in groups)  # Noncompliant {{Add parentheses around this starred generator argument.}}
#  ^^^^^^^^^^^^^^^^^^
def decorated():
    pass


def compliant(groups, values):
    f((*x for x in groups))
    f(*values)
    f(*(x for x in groups))
    f(*(x async for x in groups))
    f(x for x in groups)
    f(*(*x for x in groups))
    [*x for x in groups]
    a = (*x for x in groups)
