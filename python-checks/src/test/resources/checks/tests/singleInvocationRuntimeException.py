import unittest
import uuid
from pathlib import Path
from uuid import UUID, uuid4

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
    with pytest.raises(ValueError):  # Noncompliant {{Refactor this exception test to have only one invocation possibly throwing an exception.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        get_item().process()
#       ^^^^^^^^^^< {{Invocation possibly throwing an exception.}}
#                  ^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}


def test_pytest_with_nested_calls():
    with pytest.raises(ValueError):  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        do_work(get_item())
#       ^^^^^^^^^^^^^^^^^^^< {{Invocation possibly throwing an exception.}}
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
    with pytest.raises(ValueError):  # Noncompliant {{Refactor this exception test to have only one invocation possibly throwing an exception.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        do_work(object(1))
#       ^^^^^^^^^^^^^^^^^^< {{Invocation possibly throwing an exception.}}
#               ^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}


def test_pytest_with_unsafe_collection_arguments():
    with pytest.raises(ValueError):  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        do_work(set(5))
#       ^^^^^^^^^^^^^^^< {{Invocation possibly throwing an exception.}}
#               ^^^^^^@-1< {{Invocation possibly throwing an exception.}}


def test_pytest_with_dict_named_args():
    with pytest.raises(ValueError):
        do_work(dict(x=1))


def test_pytest_with_str_argument():
    with pytest.raises(ValueError):
        do_work(str(1))


def test_pytest_with_print():
    with pytest.raises(ValueError):
        print(do_work())


def test_pytest_with_path():
    with pytest.raises(ValueError):
        do_work(Path("missing"))


def test_pytest_with_uuid4():
    with pytest.raises(ValueError):
        do_work(uuid4())


def test_pytest_with_uuid_module():
    with pytest.raises(ValueError):
        do_work(uuid.uuid4())


def test_pytest_with_uuid_constructor():
    with pytest.raises(ValueError):
        do_work(UUID(int=0))


def test_pytest_with_len_and_repr():
    with pytest.raises(ValueError):
        do_work(len(repr(1)))


def test_pytest_with_list_and_tuple_args():
    with pytest.raises(ValueError):
        do_work(list((1,)))


def test_pytest_direct_lambda():
    pytest.raises(ValueError, lambda: get_item().process())  # Noncompliant {{Refactor this exception test to have only one invocation possibly throwing an exception.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                     ^^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}
#                                                ^^^^^^^^^@-2< {{Invocation possibly throwing an exception.}}


def test_pytest_direct_callable():
    item = get_item()
    pytest.raises(ValueError, item.process)


def test_pytest_direct_lambda_with_safe_builtin():
    pytest.raises(ValueError, lambda: do_work(str()))


def test_pytest_direct_lambda_single_invocation():
    pytest.raises(ValueError, lambda: do_work())


def test_pytest_direct_lambda_with_dict_positional_argument():
    pytest.raises(ValueError, lambda: do_work(dict(5)))


def test_pytest_direct_lambda_with_dict_named_args():
    pytest.raises(ValueError, lambda: do_work(dict(x=1)))


def test_pytest_direct_lambda_with_uuid():
    pytest.raises(ValueError, lambda: do_work(uuid4()))


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
        with self.assertRaises(ValueError):  # Noncompliant {{Refactor this exception test to have only one invocation possibly throwing an exception.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            get_item().process()
#           ^^^^^^^^^^< {{Invocation possibly throwing an exception.}}
#                      ^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}

    def test_unittest_with_safe_builtin(self):
        with self.assertRaises(ValueError):
            do_work(dict())

    def test_unittest_with_dict_named_args(self):
        with self.assertRaises(ValueError):
            do_work(dict(x=1))

    def test_unittest_with_path_and_str(self):
        with self.assertRaises(ValueError):
            do_work(str(Path("x")))

    def test_unittest_with_uuid(self):
        with self.assertRaises(ValueError):
            do_work(uuid.uuid4())

    def test_unittest_lambda(self):
        self.assertRaises(ValueError, lambda: get_item().process())  # Noncompliant {{Refactor this exception test to have only one invocation possibly throwing an exception.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                             ^^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}
#                                                        ^^^^^^^^^@-2< {{Invocation possibly throwing an exception.}}

    def test_unittest_lambda_nested_calls(self):
        self.assertRaises(ValueError, lambda: do_work(get_item()))  # Noncompliant
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                             ^^^^^^^^^^^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}
#                                                     ^^^^^^^^^^@-2< {{Invocation possibly throwing an exception.}}

    def test_unittest_lambda_with_safe_builtin(self):
        self.assertRaises(ValueError, lambda: do_work(bytearray()))

    def test_unittest_bound_method(self):
        item = get_item()
        self.assertRaises(ValueError, item.process)

    def test_unittest_lambda_single_invocation(self):
        self.assertRaises(ValueError, lambda: do_work())

    def test_unittest_lambda_with_unsafe_builtin_argument(self):
        self.assertRaises(ValueError, lambda: do_work(frozenset(5)))  # Noncompliant
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                             ^^^^^^^^^^^^^^^^^^^^^@-1< {{Invocation possibly throwing an exception.}}
#                                                     ^^^^^^^^^^^^@-2< {{Invocation possibly throwing an exception.}}

    def test_unittest_invalid_raise_method(self):
        self.assertRaisesRandom(ValueError, lambda: do_work(get_item()))


class Helper:
    def test_not_a_unittest_test_case(self):
        self.assertRaises(ValueError, lambda: do_work(get_item()))
