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

def test_except_body_with_extra_logic():
    try:
        explode()
    except ValueError as err:
        print(err)
    else:
        pytest.fail("ValueError expected")  # Noncompliant

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

def test_fail_in_except():
    try:
        explode()
    except ValueError:
        pytest.fail("unexpected")  # Noncompliant {{Remove this try/except block and let the test fail naturally if an exception is raised.}}

class SomeTest(unittest.TestCase):
    def __init__(self):
        self.user_service = None
        self.valid_user = None
        self.invalid_user = None

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

    def test_no_exception_expected(self):
        try:
            self.user_service.register_user(self.valid_user)
        except ValidationError:
            self.fail("Should not have thrown any exception")  # Noncompliant {{Remove this try/except block and let the test fail naturally if an exception is raised.}}

    def test_exception_details(self):
        try:
            self.user_service.register_user(self.invalid_user)
            self.fail("Expected ValidationError to be thrown")  # Noncompliant {{Replace this try/except block with "self.assertRaises()".}}
        except ValidationError as e:
            self.assertEqual("Invalid email", str(e))

    def test_no_issue_outside_unittest(self):
        try:
            explode()
        except ValueError:
            pass
        else:
            other.fail("ValueError expected")
