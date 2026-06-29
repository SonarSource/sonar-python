import unittest

import pytest
from pytest import fail as imported_fail

def explode():
    raise ValueError()

def test_pytest_reproducer():
    try:  # Noncompliant {{Replace this try/except block with a "pytest.raises" context manager.}}
#   ^[el=+6;ec=42]
        explode()
    except ValueError:
        pass
    else:
        pytest.fail("ValueError expected")
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Replace this fail call with a dedicated exception assertion.}}

def test_pytest_else_fail():
    try:  # Noncompliant {{Replace this try/except block with a "pytest.raises" context manager.}}
#   ^[el=+6;ec=42]
        explode()
    except ValueError:
        pass
    else:
        pytest.fail("ValueError expected")
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Replace this fail call with a dedicated exception assertion.}}

def test_pytest_try_fail():
    try:  # Noncompliant {{Replace this try/except block with a "pytest.raises" context manager.}}
#   ^[el=+6;ec=12]
        explode()
        pytest.fail("ValueError expected")
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Replace this fail call with a dedicated exception assertion.}}
    except ValueError:
        pass

def test_pytest_imported_fail():
    try:  # Noncompliant {{Replace this try/except block with a "pytest.raises" context manager.}}
#   ^[el=+6;ec=44]
        explode()
    except ValueError:
        pass
    else:
        imported_fail("ValueError expected")
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Replace this fail call with a dedicated exception assertion.}}

def test_except_body_with_extra_logic():
    try:  # Noncompliant
#   ^[el=+6;ec=42]
        explode()
    except ValueError as err:
        print(err)
    else:
        pytest.fail("ValueError expected")
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Replace this fail call with a dedicated exception assertion.}}

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

def test_no_issue_with_bare_except():
    try:
        explode()
    except:
        pytest.fail("ValueError expected")

def test_no_issue_with_except_star():
    try:
        explode()
    except* ValueError:
        pytest.fail("ValueError expected")

def test_no_issue_with_multi_statement_else():
    try:
        explode()
    except ValueError:
        pass
    else:
        message = "ValueError expected"
        pytest.fail(message)

def test_no_issue_with_multi_expression_else():
    try:
        explode()
    except ValueError:
        pass
    else:
        print("before fail"), pytest.fail("ValueError expected")

def test_no_issue_with_non_call_expression_else():
    try:
        explode()
    except ValueError:
        pass
    else:
        "ValueError expected"

def test_fail_in_except():
    try:  # Noncompliant {{Remove this try/except block and let the test fail naturally if an exception is raised.}}
#   ^[el=+4;ec=33]
        explode()
    except ValueError:
        pytest.fail("unexpected")
#       ^^^^^^^^^^^^^^^^^^^^^^^^^< {{Remove this fail call and let the test fail naturally if an exception is raised.}}

def test_no_issue_with_multi_statement_except_body():
    try:
        explode()
    except ValueError:
        message = "unexpected"
        pytest.fail(message)

def test_try_fail_with_multiple_except_clauses():
    try:  # Noncompliant {{Replace this try/except block with a "pytest.raises" context manager.}}
#   ^[el=+8;ec=37]
        explode()
        pytest.fail("ValueError expected")
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Replace this fail call with a dedicated exception assertion.}}
    except ValueError:
        pass
    except TypeError as err:
        assert "bad type" in str(err)

class SomeTest(unittest.TestCase):
    def __init__(self):
        self.user_service = None
        self.valid_user = None
        self.invalid_user = None

    def test_unittest_else_fail(self):
        try:  # Noncompliant {{Replace this try/except block with "self.assertRaises()".}}
#       ^[el=+6;ec=44]
            explode()
        except ValueError:
            pass
        else:
            self.fail("ValueError expected")
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Replace this fail call with a dedicated exception assertion.}}

    def test_unittest_try_fail(self):
        try:  # Noncompliant {{Replace this try/except block with "self.assertRaises()".}}
#       ^[el=+6;ec=16]
            explode()
            self.fail("ValueError expected")
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Replace this fail call with a dedicated exception assertion.}}
        except ValueError:
            pass

    def test_no_exception_expected(self):
        try:  # Noncompliant {{Remove this try/except block and let the test fail naturally if an exception is raised.}}
#       ^[el=+4;ec=61]
            self.user_service.register_user(self.valid_user)
        except ValidationError:
            self.fail("Should not have thrown any exception")
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Remove this fail call and let the test fail naturally if an exception is raised.}}

    def test_exception_details(self):
        try:  # Noncompliant {{Replace this try/except block with "self.assertRaises()".}}
#       ^[el=+6;ec=53]
            self.user_service.register_user(self.invalid_user)
            self.fail("Expected ValidationError to be thrown")
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Replace this fail call with a dedicated exception assertion.}}
        except ValidationError as e:
            self.assertEqual("Invalid email", str(e))

    def test_multiple_except_clauses(self):
        try:
            self.user_service.register_user(self.valid_user)
        except ValidationError:
            self.fail("got ValidationError")
        except TypeError:
            self.assertEqual("bad type", "bad type")

    def test_no_issue_outside_unittest(self):
        try:
            explode()
        except ValueError:
            pass
        else:
            other.fail("ValueError expected")
