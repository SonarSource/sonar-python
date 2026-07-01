import unittest


class TestHelper:
    def helper(self):
        return 42


class EmptyUnittestCase(unittest.TestCase):  # Noncompliant {{Add some tests to this class.}}
#     ^^^^^^^^^^^^^^^^^
    def helper(self):
        return 42
