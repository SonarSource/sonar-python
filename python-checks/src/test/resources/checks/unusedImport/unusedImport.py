from my_module import foo, bar, qix  # Noncompliant {{Remove this unused import.}}
#                          ^^^
import other_module # OK - not covered
from my_second_module import f1 as f2 # Noncompliant
#                                  ^^

from my_third_module.submodule import attr # Noncompliant
from my_foruth_module.submodule import some # OK

from __future__ import absolute_import # OK - used to instruct python interpreter
from typing import Dict # OK - it may be used in comments
from typing_extensions import Final # OK - it may be used in comments

from .. import some_symbol # Noncompliant

from some_package import symbol_to_rexport, SomeClass

__all__ = ["symbol_to_rexport"]

foo() # type: SomeClass
qix()
some()

def import_inside_function():
    from my_module import foo # Noncompliant
    print("hello")

    from my_second_module import bar # FN - redefined symbol: would need flow sensitive symbol table
    bar = 42
    print(bar)
    from my_module import def_import # Noncompliant


# This is not the same as the imported symbol above
def_import()

if True:
    from my_module import if_import # Compliant

if_import()

from sklearn.experimental import enable_iterative_imputer
from sklearn.experimental import pjdfgpojdgf as something
from sklearn.impute import IterativeImputer # Noncompliant

lazy import unused_plugin  # Noncompliant {{Remove this unused import.}}
#           ^^^^^^^^^^^^^

lazy import used_plugin
used_plugin.register()

lazy import package.submodule  # Noncompliant {{Remove this unused import.}}
#           ^^^^^^^

lazy import unused_alias as alias  # Noncompliant {{Remove this unused import.}}
#                           ^^^^^

lazy import used_alias as active
active.register()

lazy from package import unused_from  # Noncompliant {{Remove this unused import.}}
#                        ^^^^^^^^^^^
lazy from package import used_from
used_from.register()

lazy import named_in_a_comment  # OK: existing comment suppression
# named_in_a_comment

lazy import sklearn.experimental.enable_iterative_imputer as lazy_enabler  # Noncompliant
from sklearn.impute import IterativeImputer as IterativeImputerUsed
IterativeImputerUsed()


lazy from sklearn.experimental import enable_iterative_imputer as lazy_from_enabler  # Noncompliant
from sklearn.impute import IterativeImputer as LazyFromIterativeImputer
LazyFromIterativeImputer()
