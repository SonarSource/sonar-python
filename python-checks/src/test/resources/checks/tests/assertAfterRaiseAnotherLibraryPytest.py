import unittest
from tensorflow.python.platform import pytestfake

class MyTest(unittest.TestCase):
    def test_something(self):
        with self.assertRaises(ZeroDivisionError):
            assert bar() == 42 # Noncompliant {{Refactor this test; if this assertion’s argument raises an exception, the assertion will never get executed.}}

    def test_something_2(self):
        with self.assertRaises(ZeroDivisionError):
            foo()
            assert bar() == 42 # Noncompliant {{Don’t perform an assertion here; An exception is expected to be raised before its execution.}}

    # NOT pytest
    def test_non_compliant_basic():
        with pytestfake.raises(ZeroDivisionError):
            assert bar() == 42
