import copy
import os
from copy import copy as copy_fn


def noncompliant_cases():
    env = copy.copy(os.environ)  # Noncompliant {{Replace this shallow copy of "os.environ" with "os.environ.copy()".}}
    #     ^^^^^^^^^^^^^^^^^^^^^

    env = copy_fn(os.environ)  # Noncompliant
    #     ^^^^^^^^^^^^^^^^^^^

    env = copy.copy(x=os.environ)  # Noncompliant
    #     ^^^^^^^^^^^^^^^^^^^^^^^

    env = copy.copy((os.environ))  # Noncompliant
    #     ^^^^^^^^^^^^^^^^^^^^^^^

    import os as operating_system
    env = copy.copy(operating_system.environ)  # Noncompliant
    #     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def noncompliant_multiline():
    env = copy.copy((os  # Noncompliant
    #     ^[el=+2;ec=31]
                     .environ))


def compliant_cases():
    env = os.environ.copy()
    env = dict(os.environ)
    env = copy.deepcopy(os.environ)
    env = copy.copy({})
    env = copy.copy({"PATH": "/"})
    other = {"a": "b"}
    env = copy.copy(other)
    env = copy.copy(*[os.environ])
    env = copy.copy()
    env = copy.copy(os.environ, None)
    env = copy.copy(foo=os.environ)
