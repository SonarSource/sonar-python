import unittest
from uselessStatementImported import ClassWithProperty

class TestClass(unittest.TestCase):
    def test(self):
        self.assertTrue(42) # Noncompliant
        self.assertTrue(ClassWithProperty().my_property)
