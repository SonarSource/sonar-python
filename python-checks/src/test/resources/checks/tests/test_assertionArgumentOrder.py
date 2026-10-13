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
    # "actual first" is used 5 times and "expected first" 4 times, so the latter deviates from the file convention
    def test_swapped_equality_arguments(self):
        self.assertEqual(42, value())
#       Noncompliant@-1 {{Swap these 2 arguments so the actual value comes first, to match the rest of this file.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^@-1
#                        ^^@-2< {{Expected value.}}
#                            ^^^^^^^@-3< {{Actual value.}}
        self.assertEqual(first=EXPECTED_COUNT, second=value())  # Noncompliant
        self.assertIs(None, value())  # Noncompliant
        self.assertAlmostEqual(EXPECTED_PI, value())  # Noncompliant

        self.assertEqual(value(), 42)
        self.assertEqual(value(), EXPECTED_COUNT)
        self.assertIs(value(), None)
        self.assertAlmostEqual(value(), EXPECTED_PI)
        self.assertEqual(first=value(), second=42)
        self.assertEqual(value(), value())
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
