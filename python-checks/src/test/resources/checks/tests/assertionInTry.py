import unittest
import builtins

import pytest
from assertpy import assert_that


def do_something():
    raise AssertionError("Something went wrong")


def get_result():
    return "actual"


def test_pytest_assert_in_try():
    try:
        do_something()
        assert False, "Expected an AssertionError!"  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
    except AssertionError:
        pass


def test_pytest_assert_with_exception_binding():
    try:
        do_something()
        assert False  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
    except AssertionError as e:
        pass


def test_pytest_assert_except_exception():
    try:
        assert 1 == 2  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
    except Exception:
        pass


def test_pytest_assert_except_base_exception():
    try:
        assert 1 == 2  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
    except BaseException:
        pass


def test_pytest_assert_bare_except():
    try:
        assert 1 == 2  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
    except:
        pass


class MyTest(unittest.TestCase):
    def test_unittest_assert_equal(self):
        try:
            self.assertEqual(1, 2)  # Noncompliant {{Don't use assertEqual inside a try-except that catches AssertionError.}}
        except AssertionError:
            pass

    def test_unittest_assert_true(self):
        try:
            self.assertTrue(False)  # Noncompliant {{Don't use assertTrue inside a try-except that catches AssertionError.}}
        except AssertionError as e:
            pass


def test_assertpy():
    try:
        result = get_result()
        assert_that(result).is_equal_to("expected")  # Noncompliant {{Don't use is_equal_to inside a try-except that catches AssertionError.}}
    except AssertionError:
        pass


def test_assert_in_nested_block():
    try:
        if True:
            assert 1 == 2  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
    except AssertionError:
        pass


def test_nested_try_outer_swallows():
    try:
        try:
            assert 1 == 2  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
        except ValueError:
            pass
    except AssertionError:
        pass


def test_tuple_except():
    try:
        assert 1 == 2  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
    except (ValueError, AssertionError):
        pass


def test_multiple_except_clauses():
    try:
        assert 1 == 2  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
    except ValueError:
        pass
    except AssertionError:
        pass


def test_multiple_except_clauses_exception():
    try:
        assert 1 == 2  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
    except ValueError:
        pass
    except Exception:
        pass


def test_assert_that_only():
    try:
        assert_that(get_result())  # Noncompliant {{Don't use assert_that inside a try-except that catches AssertionError.}}
    except AssertionError:
        pass


def test_assertpy_described_as_chain():
    try:
        assert_that(get_result()).described_as("msg").is_equal_to("expected")  # Noncompliant {{Don't use is_equal_to inside a try-except that catches AssertionError.}}
    except AssertionError:
        pass


def test_multiple_assertions():
    try:
        assert 1 == 2  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
        assert 3 == 4  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
    except AssertionError:
        pass


def test_inner_reraise_outer_swallows():
    try:
        try:
            assert 1 == 2  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
        except AssertionError:
            raise
    except AssertionError:
        pass


def test_assert_in_for_loop():
    try:
        for _ in range(1):
            assert 1 == 2  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
    except AssertionError:
        pass


def test_qualified_assertion_error_except():
    try:
        assert 1 == 2  # Noncompliant {{Don't use assert inside a try-except that catches AssertionError.}}
    except builtins.AssertionError:
        pass


def test_compliant_pytest_raises():
    with pytest.raises(AssertionError):
        do_something()


def test_compliant_assert_outside_try():
    result = get_result()
    assert_that(result).is_equal_to("actual")


def test_compliant_except_value_error_only():
    try:
        assert 1 == 2
    except ValueError:
        pass


def test_compliant_reraise():
    try:
        assert 1 == 2
    except AssertionError:
        raise


def test_compliant_reraise_with_binding():
    try:
        assert 1 == 2
    except AssertionError as error:
        raise error


def test_compliant_assert_in_except_block():
    try:
        do_something()
    except AssertionError:
        assert True


def test_compliant_non_assert_call_in_try():
    try:
        get_result()
    except AssertionError:
        pass


def test_compliant_assert_after_try():
    try:
        do_something()
    except AssertionError:
        pass
    assert True


def test_compliant_multiple_except_only_value_error_swallows():
    try:
        raise ValueError()
    except ValueError:
        pass
    except AssertionError:
        pass


def test_compliant_star_except():
    try:
        assert 1 == 2
    except* AssertionError:
        pass


def test_compliant_inner_try_in_except_handler():
    try:
        pass
    except AssertionError:
        try:
            assert 1 == 2
        except ValueError:
            pass


class MyOtherTest(unittest.TestCase):
    def test_compliant_non_self_assert_call(self):
        try:
            unittest.TestCase.assertEqual(self, 1, 2)
        except AssertionError:
            pass
