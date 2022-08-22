import unittest
import faketest
import pytest

def external_resource_available(): return False

def foo(): return 42

def test_statement_before():
    x = 43
    if not external_resource_available():
        return  # Noncompliant {{Skip this test explicitly.}}
    if x == 42:
        assert foo() == 42

def test_something():
    if not external_resource_available():
        return  # Noncompliant {{Skip this test explicitly.}}
    else:
        assert foo() == 42

def test_unconditional_return():
    return  # Out of scope (detected by S1763)
    assert 1 != 2

def test_skip():
    if not external_resource_available():
        pytest.skip("prerequisite not met")  # OK
    else:
        assert foo() == 42

def test_ok():
    assert 1 != 2
    if not external_resource_available():
        return  # OK, assertion performed before
    else:
        assert foo() == 42

def test_some_function():
    if not external_resource_available():
        return  # OK, no assertions: might not be an actual test
    else:
        print("Hello")

def not_a_test():
    if not external_resource_available():
        return  # Not named "testXXX"
    else:
        assert foo() == 42

def helper_test():
    assert 1 != 2

# No issue should be raised if another method is called before the return (except in the "if" condition)
# As it might be a helper method performing assertions.
def test_uses_helper():
    helper_test()
    if not external_resource_available():
        return  # OK, call to "helper_test" before
    assert foo() == 42

def test_uses_variable():
    x = 42
    if not external_resource_available():
        return  # Noncompliant {{Skip this test explicitly.}}
    elif x == 43:
        x = 43
    else:
        x = 44
    assert foo() == x

def test_uses_variable_binary_left():
    x = 42
    if external_resource_available() and x == 42:
        return  # Noncompliant {{Skip this test explicitly.}}
    else:
        x = 43
    assert foo() == x

def test_uses_variable_binary_right():
    x = 42
    if x == 42 and external_resource_available():
        return  # Noncompliant {{Skip this test explicitly.}}
    else:
        x = 43
    assert foo() == x

def test_uses_variable_2():
    x = external_resource_available()
    if not x:
        return  # Accepted FN
    assert foo() == 42

def test_method_in_array():
    x = [do_assertion()]
    if whatever():
        return

def test_conditional_skip_not_first_statement():
    x = 42
    if x:
        ...
    else:
        ...
    if not external_resource_available():
        return  # Accepted FN
    assert foo() == x

# Unittest
class MyTest(unittest.TestCase):
    def test_something(self):
        if not external_resource_available():
            return  # Noncompliant {{Skip this test explicitly.}}
        else:
            "test".capitalize()
            self.assertEqual(foo(), 42)

    def test_something(self):
        if not external_resource_available():
            return
        else:
            self.assertEqual

    def test_with_skip(self):
        if not external_resource_available():
            self.skipTest("prerequisite not met")
        else:
            self.assertEqual(foo(), 42)

    def not_a_test(self):
        if not external_resource_available():
            return  # OK, method is not "testXXX"
        else:
            self.assertEqual(foo(), 42)  # Assertion doesn't matter as the method is not discoverable by the test runner
            print("hello")

    def test_no_assertions(self):
        if not external_resource_available():
            return  # OK, no assertions in the test (avoid FPs)
        else:
            foo()

class MyTestReg(unittest.TestCase):
    def test_fail_regexp_in_with(self):
        if PIL is None:
          return  # Noncompliant {{Skip this test explicitly.}}
#         ^^^^^^

        with self.assertRaisesRegexp(ValueError, '`labels` argument should be'):
          _ = ""

# Other testing library
class MyFakeTest(faketest.TestCase):
    # With an assert method which name exist in unittest
    def test_assert_existing_in_unittest(self):
        if not external_resource_available():
            return
        else:
            self.assertEqual(foo(), 42)

    # With an assert method which name does not exist in unittest
    def test_assert_not_existing_in_unittest(self):
        if not external_resource_available():
            return
        else:
            self.assertFakeEqual(foo(), 42)

# Multiple inheritance with specified class name
class TestRemoveDirectory(CommandTestMixin, unittest.TestCase):
    def test_assert_in_subfunction(self):
        if runtime.platformType == "posix":
            return # Noncompliant {{Skip this test explicitly.}}
#           ^^^^^^

        def check():
            self.assertIn({'rc': -1}, self.get_updates(), self.builder.show())
        check()

# Test heriting from the unittest framework through an additional layer of class
class customTestClass(unittest.TestCase):
    def setUp(self) -> None:
        ...
    def tearDown(self) -> None:
        ...

class inheritingFromTestClass(customTestClass):
    def test_method(self) -> None:
        if not external_resource_available():
            return  # Noncompliant {{Skip this test explicitly.}}
        else:
            self.assertEqual(foo(), 42)

# Cover edge case where classSymbol will be null (cannot be resolved : conditional definition), accepted FN
if x == 32:
    class myTestClass(unittest.TestCase):
        def testmethod(self):
            if not external_resource_available():
                return
            else:
                self.assertEqual(foo(), 42)
else:
    class myTestClass(unittest.TestCase):
        def testmethod2(self):
            if not external_resource_available():
                return
            else:
                self.assertEqual(foo(), 42)
