import unittest


class MyTest(unittest.TestCase):

    def __init__(self): ...  # OK

    def setUp(self) -> None: ...  # OK (unittest.TestCase method)

    def my_helper_method(self): ...

    def test_something(self): ...  # OK

    def test_something_else(self):
        self.my_helper_method()
        self.debug()
        ...

    def something_test(self):  # Noncompliant {{Rename this method so that it starts with "test" or remove this unused helper.}}
#       ^^^^^^^^^^^^^^
        ...

    def testSomething(self): ...  # OK

    # test must be lowercase.
    def tesT_other(self): ...  # Noncompliant {{Rename this method so that it starts with "test" or remove this unused helper.}}
#       ^^^^^^^^^^

    def inner_helper(self): ...

    def outter_helper(self):
        self.inner_helper()

    def test_helper(self):
        self.outter_helper()

    def helper(self): ...  # OK

    def unused_helper(self):  # Noncompliant
#       ^^^^^^^^^^^^^
        self.helper()

    def loop1(self):  # OK, infinite recursion should be raised by S2190
        self.loop2()

    def loop2(self):
        self.loop1()

    def outside_def(self):
        def inside(self):
            ...

        ...

    def test_outside(self):
        self.outside_def()

    def helper_used_outside(self):  # Noncompliant
        ...


def helper_used_from_mytest():
    mytest = MyTest()
    mytest.helper_used_outside()


class MyMixin(object):

    def test_something(self):
        self.assertEqual(self.some_helper(), 42)


class MyCustomTest(MyMixin, unittest.TestCase):
    """
    Classes subclassing other classes than unittest.TestCase might be mixins
    See: https://github.com/RMerl/asuswrt-merlin/blob/master/release/src/router/samba-3.6.x/lib/dnspython/tests/resolver.py
    """

    def some_helper(self):
        return 42


class MyParentTest(unittest.TestCase):
    """
    As classes subclassing unittest.TestCase will be executed as tests,
    they should define test methods and not be used as "abstract" parent helper
    We should watch out for FPs on Peach, though.
    """

    def some_helper(self):  ...  # Noncompliant


class MyChildTest(MyParentTest):

    def test_something(self):
        self.some_helper()
        ...

    def method_not_in_parent(self): # Noncompliant
        ...


class TestInfo:
    def __init__(self):
        pass

    def dumb_method(self): ...


class FunctionReference(unittest.TestCase):

    def run_ioloop(self):
        ...

    def test_ref(self):
        response = self.twisted_fetch(self.run_ioloop)
        self.assertEqual(response, 'Hello from tornado!')


class FlagsTest(unittest.TestCase):

    def setUp(self):
        self.wrapped_flags = flags._FlagValuesWrapper(self.original_flags)

    def test_func(self):
        self.assertFalse(self.wrapped_flags.is_parsed())

    def test_reference(self):
        self.wrapped_flags(['program', '--test=test'])
        self.assertEqual('test', self.wrapped_flags.test)


class CLSTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._add_databases_failures()

    @classmethod
    def _add_databases_failures(cls):
        cls.databases = ["42"]
        ...


class TestMain(unittest.TestCase):

    def get_out(self):  # Noncompliant
        ...

    def get_output(self, **options: Any) -> Any:
        return subprocess.check_output(**options)

    def test_encode_decode(self) -> None:
        output = self.get_output()
        self.assertSequenceEqual(output.splitlines(), [
            b"b'Aladdin:open sesame'",
            b"b'Aladdin:open sesame'"])

    def _not_a_test(self):
        ...

# SONARPY-1101 Fix FP on S5899 for helper methods
# We ignore helper methods which have arguments excluding the Pytest's fixtures, considering they are real helper methods not to be reported
# We also ignore methods which have some decorators
@pytest.fixture
def outFixture(): ...

class MyParentTestClass(unittest.TestCase):
    @pytest.fixture
    def parentFixture(): ...

class MyTestClass(MyParentTestClass):
    @pytest.fixture
    def inFixture(): ...

    def helper_method_with_argument(self, arg): ...

    def helper_method_with_arguments(self, arg1, arg2): ...

    def helper_method_with_only_self(self): # Noncompliant
        ...

    def helper_method_with_no_argument(): # Noncompliant
        ...

    def helper_method_with_self_and_in_fixture(self, inFixture): # Noncompliant
        ...

    def helper_method_with_self_and_in_after_fixture(self, inAfterFixture): # Noncompliant
        ...

    def helper_method_with_self_and_out_fixture(self, outFixture): # Noncompliant
        ...

    #Accepted FN : we do not check fixture coming from parent class
    def helper_method_with_self_and_parent_fixture(self, parentFixture):
        ...

    def helper_method_with_only_in_fixture(inFixture): # Noncompliant
        ...

    def helper_method_with_only_in_after_fixture(inAfterFixture): # Noncompliant
        ...

    def helper_method_with_only_out_fixture(outFixture): # Noncompliant
        ...

    def helper_method_with_self_in_fixture_and_argument(self, inFixture, arg):
        ...

    def helper_method_with_self_out_fixture_and_argument(self, outFixture, arg):
        ...

    def helper_method_with_self_parent_fixture_and_argument(self, parentFixture, arg):
        ...

    @custom_decorator
    def helper_method_with_decorator():
        ...

    @pytest.fixture
    def inAfterFixture(): ...

# Edge case
class EdgeCaseLookingLikeUnittest1(unittest.fake):
    def testMethod(): ...

class EdgeCaseLookingLikeUnittest2(fake.TestCase):
    def testMethod(): ...

from unittest import IsolatedAsyncioTestCase

class MyAsyncTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self): 
        self.data = "test_data"
    
    async def asyncTearDown(self): 
        self.data = None

    def asyncSetUp(self): 
        self.data = "test_data"

    def asyncTearDown(self): 
        self.data = None
        
    async def test_something_async(self):
        self.assertIsNotNone(self.data)
        
    async def helper_method_async(self): # Noncompliant
#             ^^^^^^^^^^^^^^^^^^^
        return "helper"
        
    async def async_setup(self): # Noncompliant
#             ^^^^^^^^^^^
        pass

class MyNonIsolatedAsyncioTest(unittest.TestCase):
    async def asyncSetUp(self): 
        self.data = "test_data"
    
    async def asyncTearDown(self): 
        self.data = None
