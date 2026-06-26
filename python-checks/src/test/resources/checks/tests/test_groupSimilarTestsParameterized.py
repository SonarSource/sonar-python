import unittest

import pytest
from parameterized import parameterized
from unittest.mock import patch


def setup_tax():
    pass


def get_tax(value):
    return value


def set_level(level):
    pass


def run_game():
    pass


def player_health():
    return 100


decorators = [lambda func: func]


def test_not_null1():  # Noncompliant {{Group these similar tests into a single parameterized test.}}
#   ^^^^^^^^^^^^^^
    setup_tax()
    assert get_tax(1) is not None


def test_not_null2():
#   ^^^^^^^^^^^^^^< {{Similar test.}}
    setup_tax()
    assert get_tax(2) is not None


def test_not_null3():
#   ^^^^^^^^^^^^^^< {{Similar test.}}
    setup_tax()
    assert get_tax(3) is not None


class TestApp:
    def test_level1(self):  # Noncompliant {{Group these similar tests into a single parameterized test.}}
#       ^^^^^^^^^^^
        set_level(1)
        run_game()
        assert player_health() == 100

    def test_level2(self):
#       ^^^^^^^^^^^< {{Similar test.}}
        set_level(2)
        run_game()
        assert player_health() == 200

    def test_level3(self):
#       ^^^^^^^^^^^< {{Similar test.}}
        set_level(3)
        run_game()
        assert player_health() == 300


class TestUnittestApp(unittest.TestCase):
    def test_user1(self):  # Noncompliant {{Group these similar tests into a single parameterized test.}}
#       ^^^^^^^^^^
        setup_tax()
        self.assertIsNotNone(get_tax(1))

    def test_user2(self):
#       ^^^^^^^^^^< {{Similar test.}}
        setup_tax()
        self.assertIsNotNone(get_tax(2))

    def test_user3(self):
#       ^^^^^^^^^^< {{Similar test.}}
        setup_tax()
        self.assertIsNotNone(get_tax(3))


@pytest.mark.parametrize("tax_id", [1, 2, 3])
def test_already_parameterized(tax_id):
    setup_tax()
    assert get_tax(tax_id) is not None


class TestParameterized(unittest.TestCase):
    @parameterized.expand([1, 2, 3])
    def test_parameterized(self, tax_id):
        self.assertIsNotNone(get_tax(tax_id))


@pytest.mark.slow
def test_decorated_but_not_parameterized():
    setup_tax()
    assert get_tax(1) is not None


def test_only_two_cases_1():
    setup_tax()
    assert get_tax(1) is not None


def test_only_two_cases_2():
    setup_tax()
    assert get_tax(2) is not None


def test_too_many_differences_1():
    set_level(1)
    run_game()
    assert player_health() == 100


def test_too_many_differences_2():
    set_level(2)
    setup_tax()
    assert player_health() == 200


def test_too_many_differences_3():
    set_level(3)
    assert get_tax(3) == 300


@decorators[0]
def test_non_standard_decorator_expression():
    setup_tax()
    assert get_tax(1) is not None


@patch("pkg.target")
def test_patched_tax1(mock_target):
    setup_tax()
    assert get_tax(1) is not None


@patch("pkg.target")
def test_patched_tax2(mock_target):
    setup_tax()
    assert get_tax(2) is not None


@patch("pkg.target")
def test_patched_tax3(mock_target):
    setup_tax()
    assert get_tax(3) is not None


class LegacySuite(unittest.TestCase):
    def test_legacy_user1(self):  # Noncompliant {{Group these similar tests into a single parameterized test.}}
#       ^^^^^^^^^^^^^^^^^
        self.assertEqual(get_tax(True), True)

    def test_legacy_user2(self):
#       ^^^^^^^^^^^^^^^^^< {{Similar test.}}
        self.assertEqual(get_tax(False), False)

    def test_legacy_user3(self):
#       ^^^^^^^^^^^^^^^^^< {{Similar test.}}
        self.assertEqual(get_tax(True), True)


def test_parameter_mismatch_1(user_id):
    setup_tax()
    assert get_tax(user_id) is not None


def test_parameter_mismatch_2(account_id):
    setup_tax()
    assert get_tax(account_id) is not None


def test_parameter_mismatch_3(customer_id):
    setup_tax()
    assert get_tax(customer_id) is not None


def test_call_with_different_arity_1():
    assert get_tax(1) == 1


def test_call_with_different_arity_2():
    assert get_tax(1, 2) == 1


def test_call_with_different_arity_3():
    assert get_tax(1, 2, 3) == 1


def test_non_literal_leaf_difference_1():
    tax = 1
    assert get_tax(tax) is not None


def test_non_literal_leaf_difference_2():
    value = 1
    assert get_tax(value) is not None


def test_non_literal_leaf_difference_3():
    amount = 1
    assert get_tax(amount) is not None


def test_string_difference_1():  # Noncompliant {{Group these similar tests into a single parameterized test.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^
    assert get_tax("alpha") is not None


def test_string_difference_2():
#   ^^^^^^^^^^^^^^^^^^^^^^^^< {{Similar test.}}
    assert get_tax("beta") is not None


def test_string_difference_3():
#   ^^^^^^^^^^^^^^^^^^^^^^^^< {{Similar test.}}
    assert get_tax("gamma") is not None


def test_literal_vs_non_literal_leaf_1():
    value = 1
    assert get_tax(1) == value


def test_literal_vs_non_literal_leaf_2():
    value = 1
    assert get_tax(account_id) == value


def test_return_variant_1():
    return


def test_return_variant_2():
    return 1


def test_return_variant_3():
    return 2


def test_raise_variant_1():
    raise ValueError("alpha")


def test_raise_variant_2():
    raise ValueError("alpha") from RuntimeError("cause")


def test_raise_variant_3():
    raise ValueError("beta")


def test_more_than_three_parameters_1():
    assert (1, "alpha", True, None) == (1, "alpha", True, None)


def test_more_than_three_parameters_2():
    assert (2, "beta", False, 1) == (2, "beta", False, 1)


def test_more_than_three_parameters_3():
    assert (3, "gamma", True, 2) == (3, "gamma", True, 2)


def test_more_than_three_literal_differences_1():
    assert get_tax(1) == ("alpha", True, None, 1)


def test_more_than_three_literal_differences_2():
    assert get_tax(2) == ("beta", False, 1, 2)


def test_more_than_three_literal_differences_3():
    assert get_tax(3) == ("gamma", True, 2, 3)


def test_repeated_parameter_difference_1():  # Noncompliant {{Group these similar tests into a single parameterized test.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    assert (1, 1, 1, 1) == (1, 1, 1, 1)


def test_repeated_parameter_difference_2():
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Similar test.}}
    assert (2, 2, 2, 2) == (2, 2, 2, 2)


def test_repeated_parameter_difference_3():
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Similar test.}}
    assert (3, 3, 3, 3) == (3, 3, 3, 3)


def test_placeholder1():
    pass


def test_placeholder2():
    pass


def test_placeholder3():
    pass


def test_not_implemented_placeholder1():
    raise NotImplementedError


def test_not_implemented_placeholder2():
    raise NotImplementedError()


def test_not_implemented_placeholder3():
    raise NotImplementedError("todo")


@pytest.mark.xfail
def test_not_implemented1():
    raise NotImplementedError


@pytest.mark.xfail
def test_not_implemented2():
    raise NotImplementedError()


@pytest.mark.xfail
def test_not_implemented3():
    raise NotImplementedError("todo")


class HelperContainer:
    def test_helper1(self):
        setup_tax()
        assert get_tax(1) is not None

    def test_helper2(self):
        setup_tax()
        assert get_tax(2) is not None

    def test_helper3(self):
        setup_tax()
        assert get_tax(3) is not None


class TestMixedKind(unittest.TestCase):
    def test_mixed_kind_1(self):
        setup_tax()
        self.assertIsNotNone(get_tax(1))


def test_mixed_kind_2():
    setup_tax()
    assert get_tax(2) is not None


def test_mixed_kind_3():
    setup_tax()
    assert get_tax(3) is not None


class TestNestedContainer:
    class TestInner:
        def test_nested_tax1(self):  # Noncompliant {{Group these similar tests into a single parameterized test.}}
#           ^^^^^^^^^^^^^^^^
            setup_tax()
            assert get_tax(1) is not None

        def test_nested_tax2(self):
#           ^^^^^^^^^^^^^^^^< {{Similar test.}}
            setup_tax()
            assert get_tax(2) is not None

        def test_nested_tax3(self):
#           ^^^^^^^^^^^^^^^^< {{Similar test.}}
            setup_tax()
            assert get_tax(3) is not None
