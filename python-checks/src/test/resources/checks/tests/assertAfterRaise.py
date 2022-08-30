import unittest
import pytest

def foo(): return 1 / 0
def bar(): return 42

# To avoid risks of FP, it is acceptable to raise issues only for simple control flows within the "with" block

# Pytest
def test_base_case_multiple_statement():
    with pytest.raises(ZeroDivisionError):
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{An exception is expected to be raised in this block.}}
        foo()
        assert bar() == 42  # Noncompliant {{Don’t perform an assertion here; An exception is expected to be raised before its execution.}}
#       ^^^^^^^^^^^^^^^^^^

def test_base_case_single_statement():
    with pytest.raises(ZeroDivisionError):
        assert bar() == 42  # Noncompliant {{Refactor this test; if this assertion’s argument raises an exception, the assertion will never get executed.}}

def test_ok():
    with pytest.raises(ZeroDivisionError):
        foo()
    assert bar() == 42  # OK

def test_base_case_2():
    with pytest.raises(ZeroDivisionError):
        assert bar() == 42  # OK, issue only on last assert
        assert foo() == 42  # Noncompliant

def test_for_loop_ok():
    with pytest.raises(ZeroDivisionError):
        for _ in range(5):
            foo()
            assert bar() == 42  # OK (might not be dead code depending on foo())

def test_for_loop_nok():
    with pytest.raises(ZeroDivisionError):
        for _ in range(5):
            foo()
        assert bar() == 42  # Noncompliant

def test_if_stmt():
    with pytest.raises(ZeroDivisionError):
        if bar() == 42:
            foo()
        else:
            assert bar() == 42  # Accepted FN

def test_if_stmt_nok():
    with pytest.raises(ZeroDivisionError):
        if bar() == 42:
            foo()
        assert bar() == 42  # Noncompliant

def test_multiple_items_1():
    with pytest.raises(ZeroDivisionError), foo():
        assert bar() == 42  # Noncompliant

def test_multiple_items_2():
    with foo(), pytest.raises(ZeroDivisionError):
        assert bar() == 42  # Noncompliant

def test_multiple_items_3():
    with foo(), bar():
        assert bar() == 42

def test_multiple_items_4():
    with pytest.raises(ZeroDivisionError), pytest.raises(TypeError):
        assert bar() == 42  # Noncompliant

def test_multiple_statements():
    with pytest.raises(ZeroDivisionError):
        a = 5, b = 3

# Edge case
def test_not_valid_assert_method():
    with pytest.random(ZeroDivisionError):
        a = 5

def test_not_pytest_lib():
    with pytestrandom.raises(ZeroDivisionError):
        a = 5

# Not raising issue in case of AssertionError raise, assuming the user is willing to write such test
def test_assert_assertion_error_unittest_1():
    error = float('nan')
    with self.assertRaises(AssertionError):
      self.assertLess(error, 1.0)

def test_assert_assertion_error_unittest_2():
    error = float('nan')
    with self.assertRaisesRegexp(AssertionError, "nan not less than 1.0"):
      self.assertLess(error, 1.0)

def test_assert_assertion_error_unittest_3():
    error = float('nan')
    with self.assertRaisesRegex(AssertionError, "nan not less than 1.0"):
      self.assertLess(error, 1.0)

def test_assert_assertion_error_unittest_4():
    error = float('nan')
    with self.assertRaisesRegex(regex="nan not less than 1.0", exception=AssertionError):
      self.assertLess(error, 1.0)

def test_assert_assertion_error_pytest_1():
    error = float('nan')
    with pytest.raises(AssertionError):
      self.assertLess(error, 1.0)

def test_assert_assertion_error_pytest_2():
    error = float('nan')
    with pytest.raises(expected_exception=AssertionError):
      self.assertLess(error, 1.0)

def test_assert_assertion_error_pytest_and_native_assert():
    error = float('nan')
    with pytest.raises(AssertionError):
      assert error < 1.0

def test_assert_assertion_error_unresolvable_error():
    with self.assertRaises(errors.OutOfRangeError):
      self.assertEqual(b"test", self.evaluate(next_element())) # Noncompliant

## Unittest
class MyTest(unittest.TestCase):
    def test_something(self):
        with self.assertRaises(ZeroDivisionError):
            self.assertEqual(foo(), 42)  # Noncompliant
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def test_something_2(self):
        with self.assertRaises(ZeroDivisionError):
            foo()
            self.assertEqual(bar(), 42)  # Noncompliant

    # Edge case
    def test_raise_callee_qualifier_not_a_name(self):
        with foo().assertRaisesRandom():
            self.assertEqual(bar(), 42)

    def test_raise_callee_qualifier_not_self(self):
        with foo.assertRaisesRandom():
            self.assertEqual(bar(), 42)

    def test_not_valid_raise_method(self):
        with self.assertRaisesRandom(ZeroDivisionError):
            self.assertEqual(bar(), 42)

    def test_not_valid_assert_method(self):
        with self.assertRaises(ZeroDivisionError):
            self.assertEqualRandom(bar(), 42)

    def test_assert_callee_qualifier_not_a_name(self):
        with self.assertRaises(ZeroDivisionError):
            foo().assertEqual(bar(), 42)

    def test_assert_callee_qualifier_not_self(self):
        with self.assertRaises(ZeroDivisionError):
            foo.assertEqual(bar(), 42)
