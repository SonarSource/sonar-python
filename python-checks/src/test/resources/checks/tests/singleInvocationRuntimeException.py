import unittest

import pytest


def get_item():
    return Item()


def do_work(value=None):
    raise ValueError(value)


class Item:
    def process(self):
        raise ValueError()


class DummyContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_pytest_with_call_chain():
    with pytest.raises(ValueError):
        get_item().process()  # Noncompliant {{Refactor this exception test to have only one invocation possibly throwing an exception.}}
#       ^^^^^^^^^^
#                  ^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}


def test_pytest_with_nested_calls():
    with pytest.raises(ValueError):
        do_work(get_item())  # Noncompliant
#       ^^^^^^^^^^^^^^^^^^^
#               ^^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}


def test_pytest_with_single_invocation():
    item = get_item()
    with pytest.raises(ValueError):
        item.process()


def test_pytest_with_safe_builtins():
    with pytest.raises(ValueError):
        do_work(str())


def test_pytest_with_safe_list():
    with pytest.raises(ValueError):
        do_work(list())


def test_pytest_with_safe_set():
    with pytest.raises(ValueError):
        do_work(set())


def test_pytest_with_safe_dict():
    with pytest.raises(ValueError):
        do_work(dict())


def test_pytest_with_safe_tuple():
    with pytest.raises(ValueError):
        do_work(tuple())


def test_pytest_with_safe_frozenset():
    with pytest.raises(ValueError):
        do_work(frozenset())


def test_pytest_with_safe_bytes():
    with pytest.raises(ValueError):
        do_work(bytes())


def test_pytest_with_safe_bytearray():
    with pytest.raises(ValueError):
        do_work(bytearray())


def test_pytest_with_safe_object():
    with pytest.raises(ValueError):
        do_work(object())


def test_pytest_with_unsafe_builtin_arguments():
    with pytest.raises(ValueError):
        do_work(object(1))  # Noncompliant {{Refactor this exception test to have only one invocation possibly throwing an exception.}}
#       ^^^^^^^^^^^^^^^^^^
#               ^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}


def test_pytest_with_unsafe_collection_arguments():
    with pytest.raises(ValueError):
        do_work(set(5))  # Noncompliant
#       ^^^^^^^^^^^^^^^
#               ^^^^^^@-1< {{Invocation possibly throwing an exception.}}


def test_pytest_direct_lambda():
    pytest.raises(ValueError, lambda: get_item().process())  # Noncompliant {{Refactor this exception test to have only one invocation possibly throwing an exception.}}
#                                     ^^^^^^^^^^
#                                                ^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}


def test_pytest_direct_callable():
    item = get_item()
    pytest.raises(ValueError, item.process)


def test_pytest_direct_lambda_with_safe_builtin():
    pytest.raises(ValueError, lambda: do_work(str()))


def test_pytest_direct_lambda_single_invocation():
    pytest.raises(ValueError, lambda: do_work())


def test_pytest_direct_lambda_with_unsafe_builtin_argument():
    pytest.raises(ValueError, lambda: do_work(dict(5)))  # Noncompliant {{Refactor this exception test to have only one invocation possibly throwing an exception.}}
#                                     ^^^^^^^^^^^^^^^^
#                                             ^^^^^^^@-1< {{Invocation possibly throwing an exception.}}


def test_non_raise_with_statement():
    with DummyContextManager():
        get_item().process()


def test_pytest_with_nested_lambda_definition():
    with pytest.raises(ValueError):
        do_work(lambda value: get_item().process())


def test_pytest_with_nested_helper_definition():
    with pytest.raises(ValueError):
        def helper():
            return get_item().process()
        do_work()


class TestCase(unittest.TestCase):
    def test_unittest_with_statement(self):
        with self.assertRaises(ValueError):
            get_item().process()  # Noncompliant {{Refactor this exception test to have only one invocation possibly throwing an exception.}}
#           ^^^^^^^^^^
#                      ^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}

    def test_unittest_with_safe_builtin(self):
        with self.assertRaises(ValueError):
            do_work(dict())

    def test_unittest_lambda(self):
        self.assertRaises(ValueError, lambda: get_item().process())  # Noncompliant {{Refactor this exception test to have only one invocation possibly throwing an exception.}}
#                                             ^^^^^^^^^^
#                                                        ^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}

    def test_unittest_lambda_nested_calls(self):
        self.assertRaises(ValueError, lambda: do_work(get_item()))  # Noncompliant
#                                             ^^^^^^^^^^^^^^^^^^^
#                                                     ^^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}

    def test_unittest_lambda_with_safe_builtin(self):
        self.assertRaises(ValueError, lambda: do_work(bytearray()))

    def test_unittest_bound_method(self):
        item = get_item()
        self.assertRaises(ValueError, item.process)

    def test_unittest_lambda_single_invocation(self):
        self.assertRaises(ValueError, lambda: do_work())

    def test_unittest_lambda_with_unsafe_builtin_argument(self):
        self.assertRaises(ValueError, lambda: do_work(frozenset(5)))  # Noncompliant
#                                             ^^^^^^^^^^^^^^^^^^^^^
#                                                     ^^^^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}

    def test_unittest_invalid_raise_method(self):
        self.assertRaisesRandom(ValueError, lambda: do_work(get_item()))


class Helper:
    def test_not_a_unittest_test_case(self):
        self.assertRaises(ValueError, lambda: do_work(get_item()))
