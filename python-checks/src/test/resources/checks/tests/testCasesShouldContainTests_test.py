import unittest


def test_module_level_test():
    assert True


class TestPytestWithoutTests:  # Noncompliant {{Add some tests to this class.}}
#     ^^^^^^^^^^^^^^^^^^^^^^
    def helper(self):
        return 42


class TestPytestWithTest:
    def test_ok(self):
        assert True


class HelperPytestClass:
    def helper(self):
        return 42


class EmptyUnittestCase(unittest.TestCase):  # Noncompliant {{Add some tests to this class.}}
#     ^^^^^^^^^^^^^^^^^
    def helper(self):
        return 42


class BaseSharedTestCase(unittest.TestCase):
    def helper(self):
        return 42


class HelperMixin(unittest.TestCase):
    def helper(self):
        return 42


class UnittestWithTest(unittest.TestCase):
    def test_ok(self):
        self.assertTrue(True)


class BaseHelperTestCase(unittest.TestCase):
    def helper(self):
        return 42


class DerivedWithOwnTest(BaseHelperTestCase):
    def test_ok(self):
        self.assertEqual(42, self.helper())


class BaseWithInheritedTest(unittest.TestCase):
    def test_from_base(self):
        self.assertTrue(True)


class DerivedInheritingTest(BaseWithInheritedTest):
    def helper(self):
        return 42


class TestBaseHelper:
    def helper(self):
        return 42


class BasePytestScaffold:
    def helper(self):
        return 42


class TestDerivedWithTest(TestBaseHelper):
    def test_ok(self):
        assert self.helper() == 42


class TestBaseWithTest:
    def test_from_base(self):
        assert True


class TestDerivedInheritingTest(TestBaseWithTest):
    def helper(self):
        return 42


class BaseWithTestAttribute(unittest.TestCase):
    test_data = True


class DerivedWithoutRealInheritedTest(BaseWithTestAttribute):  # Noncompliant {{Add some tests to this class.}}
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def helper(self):
        return 42


class TestingNameOnly(unittest.TestCase):
    def testing(self):
        return 42


class CamelCaseTestMethod(unittest.TestCase):
    def testSomething(self):
        return 42


class TestUnknownParent(UnknownParent):  # Noncompliant {{Add some tests to this class.}}
#     ^^^^^^^^^^^^^^^^^
    def helper(self):
        return 42


def TestAmbiguous():
    return 42


class TestAmbiguous:  # Noncompliant {{Add some tests to this class.}}
#     ^^^^^^^^^^^^^
    def helper(self):
        return 42


class SharedBase:
    def test_from_base(self):
        assert True


class LeftBranch(SharedBase):
    pass


class RightBranch(SharedBase):
    pass


class TestDiamondInheritance(LeftBranch, RightBranch):
    def helper(self):
        return 42


def FactoryBase():
    return object


class TestFunctionBase(FactoryBase):  # Noncompliant {{Add some tests to this class.}}
#     ^^^^^^^^^^^^^^^^
    def helper(self):
        return 42


class SharedNoTest:
    pass


class LeftNoTest(SharedNoTest):
    pass


class RightNoTest(SharedNoTest):
    pass


class TestDiamondNoTests(LeftNoTest, RightNoTest):  # Noncompliant {{Add some tests to this class.}}
#     ^^^^^^^^^^^^^^^^^^
    def helper(self):
        return 42
