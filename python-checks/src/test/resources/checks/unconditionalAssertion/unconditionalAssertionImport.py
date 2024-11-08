import unittest
from uselessStatementImported import ClassWithProperty

class TestClass(unittest.TestCase):
    def test(self):
        self.assertTrue(42) # Noncompliant
        self.assertTrue(ClassWithProperty().my_property) # OK property access that is not a constant and might have side effects
