from tensorflow.python.platform import test
import pytest

# NOT Unittest
class MyTest(test.TestCase):
    def test_something(self):
        with self.assertRaises(ZeroDivisionError):
            assert bar() == 42

    def test_something_2(self):
        with self.assertRaises(ZeroDivisionError):
            foo()
            assert bar() == 42

    def test_non_compliant_basic():
        with pytest.raises(ZeroDivisionError):
            assert bar() == 42  # Noncompliant {{Refactor this test; if this assertion’s argument raises an exception, the assertion will never get executed.}}

def test_assert_not_unittest():
    with pytest.raises(ZeroDivisionError):
        self.assertEqual(bar(), 42)

def test_assert_not_self():
    with pytest.raises(ZeroDivisionError):
        foo.assertEqual(bar(), 42)
