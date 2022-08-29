from tensorflow.python.platform import test
import unittest

# Specific test file : we are using a different test library from unittest, which should not raise issues
class OtherLibraryTest(test.TestCase):
  def test_assert_python_triggered(self):
    self.assertEqual("string", True)

class UnittestTest(unittest.TestCase):
  def test_assert_python_triggered(self):
    self.assertEqual("string", True) # Noncompliant

