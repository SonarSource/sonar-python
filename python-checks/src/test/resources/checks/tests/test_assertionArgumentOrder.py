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
        self.assertEqual(42, value())
#       Noncompliant@-1 {{Unify assertion argument order in this file; both "actual first" and "expected first" conventions are used.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^@-1
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^@-2< {{Expected value first.}}
        self.assertEqual(first=EXPECTED_COUNT, second=value())
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}
        self.assertIs(None, value())
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}
        self.assertAlmostEqual(EXPECTED_PI, value())
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}

        self.assertEqual(value(), 42)
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Actual value first.}}
        self.assertEqual(value(), value())
        self.assertIs(value(), None)
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Actual value first.}}
        self.assertAlmostEqual(value(), EXPECTED_PI)
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Actual value first.}}
        self.assertEqual(value())
        self.assertEqual(second=EXPECTED_COUNT)
        self.assertTrue(value())
        helper.assertEqual(value(), 42)

    def test_mutable_collection_actual_with_loop_expected(self):
        for expected in ([3], [3, 1], [1]):
            with self.subTest(expected=expected):
                log = []
                log.append(1)
                self.assertEqual(log, expected)

    def test_mutable_collection_append_before_assignment(self):
        def f1():
            log.append(1)

        def f2():
            log.append(2)

        def f3():
            log.append(3)

        for what, expected in (
            (f1, [1]),
            (f2, [2]),
            (f3, [3]),
        ):
            with self.subTest(what=what.__name__, expected=expected):
                log = []
                what()
                self.assertEqual(log, expected)


class Helper:
    def test_not_a_unittest_test_case(self):
        self.assertEqual(value(), 42)


def test_pytest_assertions():
    assert 42 == value()
#          ^^^^^^^^^^^^^< {{Expected value first.}}
    assert EXPECTED_COUNT == value()
#          ^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}
    assert pytest.approx(EXPECTED_PI) == value()
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}

    assert value() == 42
#          ^^^^^^^^^^^^^< {{Actual value first.}}
    assert value() == EXPECTED_COUNT
#          ^^^^^^^^^^^^^^^^^^^^^^^^^< {{Actual value first.}}
    assert value() == pytest.approx(EXPECTED_PI)
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Actual value first.}}
    assert value() == value()
    assert 42 == EXPECTED_COUNT
    assert value()
    assert 42 != value()
    assert 42 == pytest.approx(value())
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}
    assert 42 == pytest.approx(other=value())
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}


def test_assertpy_assertions():
    assert_that(42).is_equal_to(value())
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}
    assert_that(EXPECTED_COUNT).is_equal_to(value())
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}
    assert_that(42).described_as("count").is_equal_to(value())
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}
    assert_that(EXPECTED_COUNT).described_as("count").snapshot("baseline").is_equal_to(value())
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}

    assert_that(value()).is_equal_to(42)
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Actual value first.}}
    assert_that(value()).is_equal_to(EXPECTED_COUNT)
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Actual value first.}}
    assert_that(value()).described_as("count").snapshot("baseline").is_equal_to(42)
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Actual value first.}}
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


def test_mutable_collection_actual(expected):
    result = []
    assert result == expected


def test_assertion_with_message(close, expected):
    result = []
    result.append(value())
    assert result == expected, (close, result, expected)
