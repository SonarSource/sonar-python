import unittest

from assertpy import assert_that


def compute():
    return 2 + 2


def value():
    return 41 + 1


def identity(builder):
    return builder


class Holder:
    value = 42


assert compute() == compute()
assert_that(compute()).is_equal_to(compute())


class TestIdenticalArguments(unittest.TestCase):
    def noncompliant_equality(self):
        result = compute()
#                ^^^^^^^^^>
        self.assertEqual(result, result)  # Noncompliant {{Replace this assertion to not have the same actual and expected expression.}}
#                        ^^^^^^> ^^^^^^ 2
        self.assertNotEqual(result, result)  # Noncompliant
        self.assertEqual(value(), value())  # Noncompliant

    def noncompliant_identity(self):
        obj = value()
        self.assertIs(obj, obj)  # Noncompliant
        self.assertIsNot(obj, obj)  # Noncompliant

    def noncompliant_keyword_args(self):
        result = compute()
        self.assertEqual(first=result, second=result)  # Noncompliant
        self.assertIs(expr1=result, expr2=result)  # Noncompliant

    def noncompliant_attribute_access(self):
        holder = Holder()
        self.assertEqual(holder.value, holder.value)  # Noncompliant
        self.assertEqual(unknown.attr, unknown.attr)  # Noncompliant

    def compliant(self):
        result = compute()
        expected = 4
        self.assertEqual(result, expected)
        self.assertEqual(value())
        self.assertIn(result, [expected])

    def test_dynamic_call_not_checked(self):
        assertion = self.assertEqual
        assertion(value(), value())


class HelperNonTestCase:
    def test_not_a_unittest_test_case(self):
        self.assertEqual(value(), value())


def test_identical_pytest_assertions():
    actual = str(42)
#            ^^^^^^^>
    assert actual == actual  # Noncompliant {{Replace this assertion to not have the same actual and expected expression.}}
#          ^^^^^^>   ^^^^^^ 2
    assert actual != actual  # Noncompliant
    assert value() == value()  # Noncompliant
    holder = Holder()
    assert holder.value == holder.value  # Noncompliant
    assert unknown.attr == unknown.attr  # Noncompliant
    assert actual is actual
    assert value()
    assert actual == str(42)


def test_identical_assertpy_assertions():
    actual = str(42)
#            ^^^^^^^>
    assert_that(actual).is_equal_to(actual)  # Noncompliant
#               ^^^^^^>             ^^^^^^ 2
    assert_that(actual).described_as("count").is_equal_to(actual)  # Noncompliant
    assert_that(actual).is_not_equal_to(actual)  # Noncompliant
    assert_that(value()).is_equal_to(value())  # Noncompliant
    holder = Holder()
    assert_that(holder.value).is_equal_to(holder.value)  # Noncompliant
    assert_that(unknown.attr).is_equal_to(unknown.attr)  # Noncompliant

    expected = "42"
    assert_that(actual).is_equal_to(expected)
    assert_that(actual).is_equal_to()
    assert_that().is_equal_to(actual)
    builder = assert_that(actual)
    builder.is_equal_to(actual)
    identity(assert_that(actual)).is_equal_to(actual)
