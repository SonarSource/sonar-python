import unittest


class SharedTestCase(unittest.TestCase):
    def helper(self):
        return 42


class IntermediateTestCase(SharedTestCase):
    pass


class GrandchildWithTest(IntermediateTestCase):
    def test_ok(self):
        self.assertEqual(42, self.helper())
