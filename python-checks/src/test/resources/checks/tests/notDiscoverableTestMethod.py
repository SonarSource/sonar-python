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

    def method_not_in_parent(self):
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
