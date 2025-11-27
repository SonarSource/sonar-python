def bar(xs): ...


def assignement_statement():
    ls = [1, 2, 3]
    result = ls.append(42)  # Noncompliant {{Remove this use of the output from "append"; "append" doesn’t return anything.}}
#            ^^^^^^^^^^^^^
    n = ls.count(1)  # OK, count returns a integer
    return result, n


def passed_as_argument():
    ls = [1, 2, 3]
    bar(ls.sort())  # Noncompliant {{Remove this use of the output from "sort"; "sort" doesn’t return anything.}}
#       ^^^^^^^^^
    bar(xs=ls.sort())  # Noncompliant
#          ^^^^^^^^^
    bar(*ls.sort())  # This should be raised by S5633
    bar(**ls.sort())  # This should be raised by S5633
    bar(ls.copy())
    x = bar()  # OK, we don't know the type of bar


ls = [1, 2, 3]
result = ls.append(42)  # Noncompliant


def win32():
    import win32pdh
    # We should not raise issues as the stubs of win32 are inaccurate
    path = win32pdh.MakeCounterPath((None, None, None, None, -1, None)) # OK
    hc = win32pdh.AddCounter(None, path)


class SomeClass:
    def __init__(self):
        ...

class SomeOtherClass:
    ...

def foo():
    # FN: we should fallback to "type.__init__" when actual type not available
    x = SomeClass.__init__(...)  # FN
    y = SomeOtherClass.__init__(...)  # FN SONARPY-2007


def using_fcntl():
    import fcntl
    ret = fcntl.flock(..., ...)  # Noncompliant



def list_comprehensions():
    exp_list = [x() for y in unknown()]
    actual_list = [...]
    for _ in range(len(exp_list)):
        actual_list.append(smth())
    self.assertEqual(
        exp_list.sort(), # Noncompliant
        actual_list.sort() # Noncompliant
    )


def import_in_different_branch():
    if x:
        import fcntl
    def lock():
        ret = fcntl.flock(..., ...)  # Noncompliant



def smth():
    import sys
    options = trial.Options()
    options.parseOptions(["--coverage"])
    self.addCleanup(sys.settrace, sys.gettrace())
    self.assertEqual(sys.gettrace(), options.tracer.globaltrace)


def no_fp_aws_elasticloadbalancing():
    from aws_cdk.aws_elasticloadbalancingv2 import ApplicationLoadBalancer, NetworkLoadBalancer

    app_lb = ApplicationLoadBalancer()
    net_lb = NetworkLoadBalancer()

    app_listener = app_lb.add_listener(...)  # OK
    net_listener = net_lb.add_listener(...)  # OK

    print(app_listener)
    print(net_listener)


def locally_defined_function():
    def foo() -> None:
        ...

    def bar() -> int:
        return 42

    a = foo() # Noncompliant
    b = bar() # OK
