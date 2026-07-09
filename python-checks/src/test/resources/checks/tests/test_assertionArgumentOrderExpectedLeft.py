import unittest

import pytest
from assertpy import assert_that

EXPECTED_COUNT = 42
EXPECTED_PI = 3.14


def value():
    return 41 + 1


class TestUnittestAssertions(unittest.TestCase):
    def test_unittest_assertions_are_unchanged(self):
        self.assertEqual(42, value())
#       Noncompliant@-1 {{Swap these 2 arguments so they are in the correct order: actual value, expected value.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^@-1
#                        ^^@-2< {{Expected value.}}
#                            ^^^^^^^@-3< {{Actual value.}}
        self.assertEqual(value(), 42)


def test_pytest_assertions_expected_on_left():
    assert value() == 42
#   Noncompliant@-1 {{Swap these 2 sides so they are in the correct order: expected value, actual value.}}
#          ^^^^^^^^^^^^^@-1
#                     ^^@-2< {{Expected value.}}
#          ^^^^^^^@-3< {{Actual value.}}
    assert value() == EXPECTED_COUNT  # Noncompliant
    assert value() == pytest.approx(EXPECTED_PI)  # Noncompliant

    assert 42 == value()
    assert EXPECTED_COUNT == value()
    assert pytest.approx(EXPECTED_PI) == value()
    assert 42 == EXPECTED_COUNT


def test_assertpy_assertions_are_unchanged():
    assert_that(42).is_equal_to(value())
#   Noncompliant@-1 {{Pass the actual value to "assert_that" and the expected value to "is_equal_to".}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^@-1
#               ^^@-2< {{Expected value.}}
#                               ^^^^^^^@-3< {{Actual value.}}
    assert_that(value()).is_equal_to(42)
