import unittest

import pytest
from pytest import fail as imported_fail

def explode():
    raise ValueError()

def test_pytest_reproducer():
    try:
        explode()
    except ValueError:
        pass
    else:
        pytest.fail("ValueError expected")  # Noncompliant {{Replace this try/except block with a "pytest.raises" context manager.}}

def test_pytest_else_fail():
    try:
        explode()
    except ValueError:
        pass
    else:
        pytest.fail("ValueError expected")  # Noncompliant {{Replace this try/except block with a "pytest.raises" context manager.}}

def test_pytest_try_fail():
    try:
        explode()
        pytest.fail("ValueError expected")  # Noncompliant {{Replace this try/except block with a "pytest.raises" context manager.}}
    except ValueError:
        pass

def test_pytest_imported_fail():
    try:
        explode()
    except ValueError:
        pass
    else:
        imported_fail("ValueError expected")  # Noncompliant {{Replace this try/except block with a "pytest.raises" context manager.}}

def test_no_issue_when_except_is_not_empty():
    try:
        explode()
    except ValueError as err:
        print(err)
    else:
        pytest.fail("ValueError expected")

def test_no_issue_on_non_pytest_fail():
    try:
        explode()
    except ValueError:
        pass
    else:
        fail("ValueError expected")

def test_no_issue_with_finally():
    try:
        explode()
    except ValueError:
        pass
    else:
        pytest.fail("ValueError expected")
    finally:
        cleanup()

def test_no_issue_when_fail_is_in_except():
    try:
        explode()
    except ValueError:
        pytest.fail("unexpected")

class SomeTest(unittest.TestCase):
    def test_unittest_else_fail(self):
        try:
            explode()
        except ValueError:
            pass
        else:
            self.fail("ValueError expected")  # Noncompliant {{Replace this try/except block with "self.assertRaises()".}}

    def test_unittest_try_fail(self):
        try:
            explode()
            self.fail("ValueError expected")  # Noncompliant {{Replace this try/except block with "self.assertRaises()".}}
        except ValueError:
            pass

    def test_no_issue_outside_unittest(self):
        try:
            explode()
        except ValueError:
            pass
        else:
            other.fail("ValueError expected")
