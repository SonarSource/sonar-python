from fake.wrapper import unittest
from fake2.mywrapper import pytest

# Pytest
def test_base_case_multiple_statement():
    with pytest.raises(ZeroDivisionError):
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>
        foo()
        assert bar() == 42  # Noncompliant {{Don’t perform an assertion here; An exception is expected to be raised before its execution.}}
#       ^^^^^^^^^^^^^^^^^^

# Unittest
class MyTest(unittest.TestCase):
    def test_something(self):
        with self.assertRaises(ZeroDivisionError):
            self.assertEqual(foo(), 42)  # Noncompliant
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def test_something_2(self):
        with self.assertRaises(ZeroDivisionError):
            foo()
            self.assertEqual(bar(), 42)  # Noncompliant
