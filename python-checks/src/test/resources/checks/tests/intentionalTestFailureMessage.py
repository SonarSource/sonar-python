import pytest
from pytest import fail as imported_fail


def test_assert_false():
    assert False  # Noncompliant {{Replace this assertion with pytest.fail(...) and provide a message.}}
#   ^^^^^^^^^^^^


def test_assert_zero():
    assert 0  # Noncompliant {{Replace this assertion with pytest.fail(...) and provide a message.}}
#   ^^^^^^^^


def test_assert_false_with_message():
    assert False, "not ready yet"  # Noncompliant {{Replace this assertion with pytest.fail(...) and provide a message.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def test_assert_zero_parenthesized():
    assert (0)  # Noncompliant {{Replace this assertion with pytest.fail(...) and provide a message.}}
#   ^^^^^^^^^^


def test_assert_equality_ok():
    assert x == 1


def test_assert_name_ok():
    assert some_name


def test_assert_true_ok():
    assert True


def test_assert_one_ok():
    assert 1


def test_assert_non_integral_float_ok():
    # valueAsLong throws NumberFormatException for non-integral floats
    assert 0.5


def test_assert_zero_float():
    assert 0.0  # Noncompliant {{Replace this assertion with pytest.fail(...) and provide a message.}}
#   ^^^^^^^^^^


def test_pytest_fail_no_args():
    pytest.fail()  # Noncompliant {{Add a message explaining why this test fails.}}
#   ^^^^^^^^^^^^^


def test_pytest_fail_empty_message():
    pytest.fail("")  # Noncompliant {{Add a message explaining why this test fails.}}
#   ^^^^^^^^^^^^^^^


def test_pytest_fail_empty_reason_keyword():
    pytest.fail(reason="")  # Noncompliant {{Add a message explaining why this test fails.}}
#   ^^^^^^^^^^^^^^^^^^^^^^


def test_pytest_fail_whitespace_message():
    pytest.fail("   ")  # Noncompliant {{Add a message explaining why this test fails.}}
#   ^^^^^^^^^^^^^^^^^^


def test_pytest_fail_whitespace_reason_keyword():
    pytest.fail(reason="   ")  # Noncompliant {{Add a message explaining why this test fails.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^


def test_pytest_fail_with_message_ok():
    pytest.fail("feature not implemented yet")


def test_pytest_fail_with_reason_keyword_ok():
    pytest.fail(reason="blocked on issue 42")


def test_pytest_fail_with_legacy_msg_keyword_ok():
    pytest.fail(msg="Sample check failed")


def test_pytest_fail_with_legacy_msg_keyword_and_pytrace_ok():
    pytest.fail(msg="Sample check failed", pytrace=False)


def test_pytest_fail_with_non_string_legacy_msg_ok(message):
    pytest.fail(msg=message)


def test_pytest_fail_empty_legacy_msg_keyword():
    pytest.fail(msg="")  # Noncompliant {{Add a message explaining why this test fails.}}
#   ^^^^^^^^^^^^^^^^^^^


def test_pytest_fail_whitespace_legacy_msg_keyword():
    pytest.fail(msg="   ")  # Noncompliant {{Add a message explaining why this test fails.}}
#   ^^^^^^^^^^^^^^^^^^^^^^


def test_pytest_fail_with_non_string_reason_ok(message):
    pytest.fail(reason=message)


def test_pytest_fail_with_non_string_message_ok(message):
    pytest.fail(message)


def test_imported_fail_no_args():
    imported_fail()  # Noncompliant {{Add a message explaining why this test fails.}}
#   ^^^^^^^^^^^^^^^


def test_local_fail_ok():
    def fail():
        pass
    fail()


class SomeTest:
    def test_unittest_fail_ok(self):
        self.fail()
