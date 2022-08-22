from fake.wrapper import unittest

class TestRemoveDirectory(CommandTestMixin, unittest.TestCase):
    def test_simple_exception(self):
        if runtime.platformType == "posix":
            return # Noncompliant {{Skip this test explicitly.}}

        self.assertIn({'rc': -1}, self.get_updates(), self.builder.show())

class TestCopyDirectory(CommandTestMixin, unittest.TestCase):
    def test_simple_exception(self):
        if runtime.platformType == "posix":
            return # Noncompliant {{Skip this test explicitly.}}

        self.assertIn({'rc': -1}, self.get_updates(), self.builder.show())
