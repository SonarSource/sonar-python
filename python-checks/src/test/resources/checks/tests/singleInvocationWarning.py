import copy
import uuid
import warnings
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest
from numpy import eye, identity, ones, zeros
from numpy.polynomial.legendre import legval
from scipy.special import legendre


def get_item():
    return Item()


def setup():
    pass


def do_work(value=None):
    warnings.warn("x", UserWarning)


class Item:
    def process(self):
        warnings.warn("x", UserWarning)


class DummyContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_pytest_with_call_chain():
    with pytest.warns(UserWarning):  # Noncompliant {{Refactor this warning test to have only one invocation possibly emitting a warning.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        get_item().process()
#       ^^^^^^^^^^< {{Invocation possibly emitting a warning.}}
#                  ^^^^^^^^^@-1< {{Invocation possibly emitting a warning.}}


def test_pytest_with_nested_calls():
    with pytest.warns(UserWarning):  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        do_work(get_item())
#       ^^^^^^^^^^^^^^^^^^^< {{Invocation possibly emitting a warning.}}
#               ^^^^^^^^^^@-1< {{Invocation possibly emitting a warning.}}


def test_pytest_with_multiple_statements():
    with pytest.warns(UserWarning):  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        setup()
#       ^^^^^^^< {{Invocation possibly emitting a warning.}}
        warnings.warn("x", UserWarning)
#                ^^^^^^^^^^^^^^^^^^^^^^< {{Invocation possibly emitting a warning.}}


def test_pytest_with_single_invocation():
    item = get_item()
    with pytest.warns(UserWarning):
        item.process()


def test_pytest_setup_outside():
    setup()
    with pytest.warns(UserWarning):
        warnings.warn("x", UserWarning)


def test_pytest_with_safe_builtins():
    with pytest.warns(UserWarning):
        do_work(str())


def test_pytest_with_safe_list():
    with pytest.warns(UserWarning):
        do_work(list())


def test_pytest_with_safe_set():
    with pytest.warns(UserWarning):
        do_work(set())


def test_pytest_with_safe_dict():
    with pytest.warns(UserWarning):
        do_work(dict())


def test_pytest_with_safe_tuple():
    with pytest.warns(UserWarning):
        do_work(tuple())


def test_pytest_with_safe_frozenset():
    with pytest.warns(UserWarning):
        do_work(frozenset())


def test_pytest_with_safe_bytes():
    with pytest.warns(UserWarning):
        do_work(bytes())


def test_pytest_with_safe_bytearray():
    with pytest.warns(UserWarning):
        do_work(bytearray())


def test_pytest_with_safe_object():
    with pytest.warns(UserWarning):
        do_work(object())


def test_pytest_with_unsafe_builtin_arguments():
    with pytest.warns(UserWarning):  # Noncompliant {{Refactor this warning test to have only one invocation possibly emitting a warning.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        do_work(object(1))
#       ^^^^^^^^^^^^^^^^^^< {{Invocation possibly emitting a warning.}}
#               ^^^^^^^^^@-1< {{Invocation possibly emitting a warning.}}


def test_pytest_with_unsafe_collection_arguments():
    with pytest.warns(UserWarning):  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        do_work(set(5))
#       ^^^^^^^^^^^^^^^< {{Invocation possibly emitting a warning.}}
#               ^^^^^^@-1< {{Invocation possibly emitting a warning.}}


def test_pytest_with_dict_named_args():
    with pytest.warns(UserWarning):
        do_work(dict(x=1))


def test_pytest_with_str_argument():
    with pytest.warns(UserWarning):
        do_work(str(1))


def test_pytest_with_print():
    with pytest.warns(UserWarning):
        print(do_work())


def test_pytest_with_path():
    with pytest.warns(UserWarning):
        do_work(Path("missing"))


def test_pytest_with_uuid4():
    with pytest.warns(UserWarning):
        do_work(uuid4())


def test_pytest_with_uuid_module():
    with pytest.warns(UserWarning):
        do_work(uuid.uuid4())


def test_pytest_with_uuid_constructor():
    with pytest.warns(UserWarning):
        do_work(UUID(int=0))


def test_pytest_with_numpy_zeros():
    with pytest.warns(UserWarning):
        do_work(np.zeros(3))


def test_pytest_with_numpy_ones():
    with pytest.warns(UserWarning):
        do_work(ones((2, 2)))


def test_pytest_with_numpy_identity():
    with pytest.warns(UserWarning):
        do_work(identity(3))


def test_pytest_with_numpy_eye():
    with pytest.warns(UserWarning):
        do_work(eye(3))


def test_pytest_with_numpy_empty():
    with pytest.warns(UserWarning):
        do_work(np.empty(2))


def test_pytest_with_numpy_full():
    with pytest.warns(UserWarning):
        do_work(np.full(2, 0))


def test_pytest_with_numpy_zeros_like():
    with pytest.warns(UserWarning):
        do_work(np.zeros_like([1, 2]))


def test_pytest_with_numpy_ones_like():
    with pytest.warns(UserWarning):
        do_work(np.ones_like([1, 2]))


def test_pytest_with_numpy_empty_like():
    with pytest.warns(UserWarning):
        do_work(np.empty_like([1, 2]))


def test_pytest_with_numpy_full_like():
    with pytest.warns(UserWarning):
        do_work(np.full_like([1, 2], 0))


def test_pytest_with_numpy_array():
    with pytest.warns(UserWarning):
        do_work(np.array([1, 2]))


def test_pytest_with_numpy_asarray():
    with pytest.warns(UserWarning):
        do_work(np.asarray([1, 2]))


def test_pytest_with_numpy_arange():
    with pytest.warns(UserWarning):
        do_work(np.arange(3))


def test_pytest_with_numpy_linspace():
    with pytest.warns(UserWarning):
        do_work(np.linspace(0, 1, 5))


def test_pytest_with_numpy_random_randn():
    with pytest.warns(UserWarning):
        do_work(np.random.randn(2, 2))


def test_pytest_with_numpy_random_random():
    with pytest.warns(UserWarning):
        do_work(np.random.random((2, 2)))


def test_pytest_with_numpy_random_rand():
    with pytest.warns(UserWarning):
        do_work(np.random.rand(2, 2))


def test_pytest_with_numpy_random_randint():
    with pytest.warns(UserWarning):
        do_work(np.random.randint(0, 10, size=3))


def test_pytest_with_numpy_random_default_rng():
    with pytest.warns(UserWarning):
        do_work(np.random.default_rng(0))


def test_pytest_with_numpy_random_state():
    with pytest.warns(UserWarning):
        do_work(np.random.RandomState(0))


def test_pytest_with_numpy_copy():
    with pytest.warns(UserWarning):
        do_work(np.copy([1, 2]))


def test_pytest_with_numpy_legendre():
    with pytest.warns(UserWarning):
        do_work(legval(0.5, [1, 0, 1]))


def test_pytest_with_scipy_legendre():
    with pytest.warns(UserWarning):
        do_work(legendre(3))


def test_pytest_with_copy_copy():
    with pytest.warns(UserWarning):
        do_work(copy.copy([1, 2]))


def test_pytest_with_copy_deepcopy():
    with pytest.warns(UserWarning):
        do_work(copy.deepcopy([1, 2]))


def test_pytest_with_len_and_repr():
    with pytest.warns(UserWarning):
        do_work(len(repr(1)))


def test_pytest_with_list_and_tuple_args():
    with pytest.warns(UserWarning):
        do_work(list((1,)))


def test_non_warns_with_statement():
    with DummyContextManager():
        get_item().process()


def test_pytest_raises_is_out_of_scope():
    with pytest.raises(ValueError):
        get_item().process()


def test_pytest_with_nested_lambda_definition():
    with pytest.warns(UserWarning):
        do_work(lambda value: get_item().process())


def test_pytest_with_nested_helper_definition():
    with pytest.warns(UserWarning):
        def helper():
            return get_item().process()
        do_work()


def test_pytest_warns_from_import():
    from pytest import warns
    with warns(UserWarning):  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^
        get_item().process()
#       ^^^^^^^^^^< {{Invocation possibly emitting a warning.}}
#                  ^^^^^^^^^@-1< {{Invocation possibly emitting a warning.}}
