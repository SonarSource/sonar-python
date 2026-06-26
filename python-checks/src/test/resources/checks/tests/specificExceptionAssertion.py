import unittest

import pytest
from pytest import raises as imported_raises


def explode():
    raise ValueError("bad value")


class CustomException(Exception):
    pass


def raise_custom_exception():
    raise CustomException("custom")


class DummyContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


dummy_context_manager = DummyContextManager()


def test_pytest_raises_exception():
    with pytest.raises(Exception):  # Noncompliant {{Specify a more specific exception type here.}}
#                      ^^^^^^^^^
        explode()


def test_pytest_raises_base_exception():
    with pytest.raises(BaseException):  # Noncompliant {{Specify a more specific exception type here.}}
#                      ^^^^^^^^^^^^^
        explode()


def test_pytest_raises_expected_exception_keyword():
    with pytest.raises(expected_exception=Exception):  # Noncompliant {{Specify a more specific exception type here.}}
#                                         ^^^^^^^^^
        explode()


def test_pytest_imported_raises():
    with imported_raises(Exception):  # Noncompliant {{Specify a more specific exception type here.}}
#                        ^^^^^^^^^
        explode()


def test_pytest_raises_specific_exception():
    with pytest.raises(ValueError):
        explode()


def test_pytest_raises_custom_exception():
    with pytest.raises(CustomException):
        raise_custom_exception()


def test_pytest_raises_generic_exception_with_match():
    with pytest.raises(Exception, match="bad value"):
        explode()


def test_pytest_direct_raises_exception():
    pytest.raises(Exception, explode)  # Noncompliant {{Specify a more specific exception type here.}}
#                 ^^^^^^^^^


def test_pytest_direct_imported_raises_exception():
    imported_raises(BaseException, explode)  # Noncompliant {{Specify a more specific exception type here.}}
#                   ^^^^^^^^^^^^^


def test_with_non_call_context_manager():
    with dummy_context_manager:
        explode()


def test_raise_qualifier_named_helper_parameter(helper):
    with helper.assertRaises(Exception):
        explode()


def test_raise_qualifier_named_cls_parameter(cls):
    with cls.assertRaises(Exception):
        explode()


class MyTest(unittest.TestCase):
    def test_unittest_assert_raises_exception(self):
        with self.assertRaises(Exception):  # Noncompliant {{Specify a more specific exception type here.}}
#                              ^^^^^^^^^
            explode()

    def test_unittest_assert_raises_base_exception(self):
        with self.assertRaises(BaseException):  # Noncompliant {{Specify a more specific exception type here.}}
#                              ^^^^^^^^^^^^^
            explode()

    def test_unittest_assert_raises_regex_exception(self):
        with self.assertRaisesRegex(Exception, "bad"):
            explode()

    def test_unittest_assert_raises_regexp_exception(self):
        with self.assertRaisesRegexp(Exception, "bad"):
            explode()

    def test_unittest_assert_raises_keyword_exception(self):
        with self.assertRaisesRegex(exception=Exception, regex="bad"):
            explode()

    def test_unittest_assert_raises_specific_exception(self):
        with self.assertRaisesRegex(ValueError, "bad value"):
            explode()

    def test_unittest_assert_raises_custom_exception(self):
        with self.assertRaises(CustomException):
            raise_custom_exception()

    def test_unittest_direct_assert_raises_exception(self):
        self.assertRaises(Exception, explode)  # Noncompliant {{Specify a more specific exception type here.}}
#                         ^^^^^^^^^

    def test_unittest_direct_assert_raises_regex_exception(self):
        self.assertRaisesRegex(Exception, "bad", explode)

    def test_unittest_direct_assert_raises_specific_exception(self):
        self.assertRaises(ValueError, explode)

    def test_unittest_raise_qualifier_not_self(self):
        with helper.assertRaises(Exception):
            explode()

    def test_unittest_raise_qualifier_not_a_name(self):
        with helper().assertRaises(Exception):
            explode()
