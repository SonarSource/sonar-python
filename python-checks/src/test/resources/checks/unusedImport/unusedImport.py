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

foo()
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
