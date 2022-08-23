from tensorflow.python.platform import test

# Specific test file : we are using a different test library from unittest, which should not raise issues
class ExceptionsTest(test.TestCase):
  def test_assert_python_triggered(self):
    if not __debug__:
      return

    self.assertEqual("5", 5)

  def test_mandatory_issue(self):
    x = 5
    if x == 5:
        return # Noncompliant {{Skip this test explicitly.}}
    assert x != 5
