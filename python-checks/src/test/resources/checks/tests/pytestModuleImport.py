from pytest import mark, raises  # Noncompliant {{Import "pytest" as a module.}}
#^[sc=1;ec=31]
from pytest import approx as approximately  # Noncompliant
#^[sc=1;ec=42]
from pytest import *  # Noncompliant
#^[sc=1;ec=20]
from pytest.foo import bar  # Noncompliant
#^[sc=1;ec=26]
import pytest as pt  # Noncompliant {{Do not alias the "pytest" module.}}
#      ^^^^^^^^^^^^
import os, pytest as pt2  # Noncompliant
#          ^^^^^^^^^^^^^
import pytest.foo as pf  # Noncompliant
#      ^^^^^^^^^^^^^^^^

import pytest
import pytest.foo
import os, pytest
import pytest as pytest  # explicit re-export, no renaming

import pytest_asyncio as pytest_async
from pytest_mock import MockerFixture
from _pytest.monkeypatch import MonkeyPatch
from pytest_django.asserts import assertNumQueries
from mypytest import something
from unittest import mock


def local_imports():
    from pytest import fixture  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^
    import pytest as inner  # Noncompliant
#          ^^^^^^^^^^^^^^^
    import pytest
    from . import pytest
    from .pytest import helper
    from ..pytest.utils import helper2
