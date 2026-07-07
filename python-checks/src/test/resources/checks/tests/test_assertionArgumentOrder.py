import unittest

import pytest
from assertpy import assert_that

EXPECTED_COUNT = 42
EXPECTED_PI = 3.14

assert 42 == value()  # OK, not inside a pytest-style test function
assert_that(42).is_equal_to(value())  # OK, not inside a supported test function


def value():
    return 41 + 1


class TestUnittestAssertions(unittest.TestCase):
    def test_swapped_equality_arguments(self):
        self.assertEqual(42, value())  # Noncompliant {{Swap these 2 arguments so they are in the correct order: actual value, expected value.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 0
        self.assertEqual(first=EXPECTED_COUNT, second=value())  # Noncompliant
        self.assertIs(None, value())  # Noncompliant
        self.assertAlmostEqual(EXPECTED_PI, value())  # Noncompliant

        self.assertEqual(value(), 42)
        self.assertEqual(value(), value())
        self.assertIs(value(), None)
        self.assertAlmostEqual(value(), EXPECTED_PI)
        self.assertEqual(value())
        self.assertEqual(second=EXPECTED_COUNT)
        self.assertTrue(value())
        helper.assertEqual(value(), 42)


class Helper:
    def test_not_a_unittest_test_case(self):
        self.assertEqual(value(), 42)


def test_pytest_assertions():
    assert 42 == value()  # Noncompliant {{Swap these 2 sides so they are in the correct order: actual value, expected value.}}
#          ^^^^^^^^^^^^^ 0
    assert EXPECTED_COUNT == value()  # Noncompliant
    assert pytest.approx(EXPECTED_PI) == value()  # Noncompliant

    assert value() == 42
    assert value() == EXPECTED_COUNT
    assert value() == pytest.approx(EXPECTED_PI)
    assert value() == value()
    assert 42 == EXPECTED_COUNT
    assert value()
    assert 42 != value()
    assert 42 == pytest.approx(value())  # Noncompliant
    assert 42 == pytest.approx(other=value())  # Noncompliant


def test_assertpy_assertions():
    assert_that(42).is_equal_to(value())  # Noncompliant {{Pass the actual value to "assert_that" and the expected value to "is_equal_to".}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 0
    assert_that(EXPECTED_COUNT).is_equal_to(value())  # Noncompliant
    assert_that(42).described_as("count").is_equal_to(value())  # Noncompliant
    assert_that(EXPECTED_COUNT).described_as("count").snapshot("baseline").is_equal_to(value())  # Noncompliant

    assert_that(value()).is_equal_to(42)
    assert_that(value()).is_equal_to(EXPECTED_COUNT)
    assert_that(value()).described_as("count").snapshot("baseline").is_equal_to(42)
    assert_that(value()).is_equal_to(value())
    assert_that(42).is_equal_to(EXPECTED_COUNT)
    assert_that(value()).is_equal_to()
    assert_that().is_equal_to(value())
    assert_that(value()).is_not_equal_to(42)
    builder.is_equal_to(42)
    factory().is_equal_to(42)
    other_call(value()).is_equal_to(42)


def helper_assertions():
    assert 42 == value()
    assert_that(42).is_equal_to(value())
