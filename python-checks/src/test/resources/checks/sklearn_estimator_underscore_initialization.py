from sklearn.base import BaseEstimator, RegressorMixin

class MyEstimator(BaseEstimator):
     #^^^^^^^^^^^> {{The attribute is used in this estimator}}
    def __init__(self) -> None:
        self.a_ = None # Noncompliant {{Move this estimated attribute in the `fit` method.}}
        #    ^^
        self.something_ok = True
        self.__something_private__ = True
        local_variable = 5.
        callable().a_  = []

    def another_method(self):
        self.b_ = True

    def fit(x, y):
        self.a_ = 5

# def CustomRegressor

class UnrelatedClass():
    def __init__(self) -> None:
        self.a_ = None

class InheritingEstimator(MyEstimator):
    def __init__(self) -> None:
        self.a_ = None # Noncompliant
        #    ^^

class AnotherUnrelatedClass(UnrelatedClass):
    def __init__(self) -> None:
        self.a_ = None

class DirectlyFromMixin(RegressorMixin):
    def __init__(self) -> None:
        self.a_ = None # Noncompliant

class IndirectlyFromMixinStep1():
    ...
class IndirectlyFromMixinStep2(RegressorMixin):
    ...
class IndirectlyFromMixinStep3(IndirectlyFromMixinStep2):
    ...

class IndirectlyFromMixin(IndirectlyFromMixinStep1, IndirectlyFromMixinStep3):
    def __init__(self) -> None:
        self.a_ = None # Noncompliant

def __init__():
    ...

class Duplicated(BaseEstimator):
    def __init__(self) -> None:
        self.something = None

class Duplicated(BaseEstimator):
    def __init__(self) -> None:
        self.something = True

class TestQuickFix(BaseEstimator):
    def __init__(self):
        self.a_, self.b_ = None, True # Noncompliant
        # Noncompliant@-1
        self.c_ = True # Noncompliant
        (self.d_, self.e_) = None, True # Noncompliant
        # Noncompliant@-1
        self.f_, (self.g_, self.h_) = None, True, None # Noncompliant
        # Noncompliant@-1
        # Noncompliant@-2
import glob
import os
import os.path
import sys
import traceback
import argparse
from collections import defaultdict

from typing import (
    List, Dict, Tuple, Iterable, Mapping, Optional, Set, cast,
)
from typing_extensions import Final

import mypy.build
import mypy.parse
import mypy.errors
import mypy.traverser
import mypy.mixedtraverser
import mypy.util
from mypy import defaults
from mypy.modulefinder import (
    ModuleNotFoundReason, FindModuleCache, SearchPaths, BuildSource, default_lib_path
)
from mypy.nodes import (
    Expression, IntExpr, UnaryExpr, StrExpr, BytesExpr, NameExpr, FloatExpr, MemberExpr,
    TupleExpr, ListExpr, ComparisonExpr, CallExpr, IndexExpr, EllipsisExpr,
    ClassDef, MypyFile, Decorator, AssignmentStmt, TypeInfo,
    IfStmt, ImportAll, ImportFrom, Import, FuncDef, FuncBase, TempNode, Block,
    Statement, OverloadedFuncDef, ARG_POS, ARG_STAR, ARG_STAR2, ARG_NAMED, ARG_NAMED_OPT
)
from mypy.stubgenc import generate_stub_for_c_module
from mypy.stubutil import (
    default_py2_interpreter, CantImport, generate_guarded,
    walk_packages, find_module_path_and_all_py2, find_module_path_and_all_py3,
    report_missing, fail_missing, remove_misplaced_type_comments, common_dir_prefix
)
from mypy.stubdoc import parse_all_signatures, find_unique_signatures, Sig
from mypy.options import Options as MypyOptions
from mypy.types import (
    Type, TypeStrVisitor, CallableType, UnboundType, NoneType, TupleType, TypeList, Instance,
    AnyType
)
from mypy.visitor import NodeVisitor
from mypy.find_sources import create_source_list, InvalidSourceList
from mypy.build import build
from mypy.errors import CompileError, Errors
from mypy.traverser import has_return_statement
from mypy.moduleinspect import ModuleInspect

class StubGenerator(mypy.traverser.TraverserVisitor):
    """Generate stub text from a mypy AST."""

    def __init__(self,
                 _all_: Optional[List[str]], pyversion: Tuple[int, int],
                 include_private: bool = False,
                 analyzed: bool = False,
                 export_less: bool = False) -> None:
        # Best known value of __all__.
        self._all_ = _all_
        self._output = []  # type: List[str]
        self._decorators = []  # type: List[str]
        self._import_lines = []  # type: List[str]
        # Current indent level (indent is hardcoded to 4 spaces).
        self._indent = ''
        # Stack of defined variables (per scope).
        self._vars = [[]]  # type: List[List[str]]
        # What was generated previously in the stub file.
        self._state = EMPTY
        self._toplevel_names = []  # type: List[str]
        self._pyversion = pyversion
        self._include_private = include_private
        self.import_tracker = ImportTracker()
        # Was the tree semantically analysed before?
        self.analyzed = analyzed
        # Disable implicit exports of package-internal imports?
        self.export_less = export_less
        # Add imports that could be implicitly generated
        self.import_tracker.add_import_from("collections", [("namedtuple", None)])
        # Names in __all__ are required
        for name in _all_ or ():
            if name not in IGNORED_DUNDERS:
                self.import_tracker.reexport(name)
        self.defined_names = set()  # type: Set[str]
        # Short names of methods defined in the body of the current class
        self.method_names = set()  # type: Set[str]

    def visit_mypy_file(self, o: MypyFile) -> None:
        self.module = o.fullname  # Current module being processed
        self.path = o.path
        self.defined_names = find_defined_names(o)
        self.referenced_names = find_referenced_names(o)
        typing_imports = ["Any", "Optional", "TypeVar"]
        for t in typing_imports:
            if t not in self.defined_names:
                alias = None
            else:
                alias = '_' + t
            self.import_tracker.add_import_from("typing", [(t, alias)])
        super().visit_mypy_file(o)
        undefined_names = [name for name in self._all_ or []
                           if name not in self._toplevel_names]
        if undefined_names:
            if self._state != EMPTY:
                self.add('\n')
            self.add('# Names in __all__ with no definition:\n')
            for name in sorted(undefined_names):
                self.add('#   %s\n' % name)
