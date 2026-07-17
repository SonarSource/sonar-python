import pytest


def double(operand):
    return operand * 2


@pytest.mark.parametrize('operand,expected', [])  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                            ^^
def test_double_empty_list(operand, expected):
    assert double(operand) == expected


@pytest.mark.parametrize('operand,expected', ())  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                            ^^
def test_double_empty_tuple(operand, expected):
    assert double(operand) == expected


@pytest.mark.parametrize('operand,expected', [
    (1, 2),
    (2, 4),
])
def test_double_populated_list(operand, expected):
    assert double(operand) == expected


@pytest.mark.parametrize('operand,expected', (
    (1, 2),
    (2, 4),
))
def test_double_non_empty_tuple(operand, expected):
    assert double(operand) == expected


empty_cases = []


@pytest.mark.parametrize('operand', empty_cases)  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^^^^^^^^^^
def test_empty_list_variable_never_appended(operand):
    assert double(operand) == operand * 2


populated_before = []
populated_before.append(1)
populated_before.append(2)


@pytest.mark.parametrize('operand', populated_before)
def test_list_populated_via_append_before(operand):
    assert double(operand) == operand * 2


populated_after = []


@pytest.mark.parametrize('operand', populated_after)
def test_list_populated_via_append_after(operand):
    assert double(operand) == operand * 2


populated_after.append(1)


populated_via_extend = []
populated_via_extend.extend([(1, 2)])


@pytest.mark.parametrize('case', populated_via_extend)
def test_list_populated_via_extend(case):
    assert double(case[0]) == case[1]


populated_via_iadd = []
populated_via_iadd += [(1, 2)]


@pytest.mark.parametrize('case', populated_via_iadd)
def test_list_populated_via_iadd(case):
    assert double(case[0]) == case[1]


populated_via_insert = []
populated_via_insert.insert(0, (1, 2))


@pytest.mark.parametrize('case', populated_via_insert)
def test_list_populated_via_insert(case):
    assert double(case[0]) == case[1]


wrong_method = []
wrong_method.count(0)


@pytest.mark.parametrize('operand', wrong_method)  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^^^^^^^^^^^
def test_list_with_non_populating_method_call(operand):
    assert operand is None


dict_cases = {}
dict_cases.update({'a': 1})


@pytest.mark.parametrize('case', dict_cases)
def test_dict_populated_via_update(case):
    assert case == 'a'


setdefault_cases = {}
setdefault_cases.setdefault('a', 1)


@pytest.mark.parametrize('case', setdefault_cases)
def test_dict_populated_via_setdefault(case):
    assert case == 'a'


merged_dict_cases = {}
merged_dict_cases |= {'a': 1}


@pytest.mark.parametrize('case', merged_dict_cases)
def test_dict_populated_via_ior(case):
    assert case == 'a'


class TestOffsets:
    offset_cases = []
    offset_cases.append(
        (1, 2),
    )
    offset_cases.append(
        (2, 4),
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        operand, expected = case
        assert double(operand) == expected


class TestClassCases:
    class_cases = []
    class_cases.append(1)

    @pytest.mark.parametrize('case', class_cases)
    def test_class_cases(self, case):
        assert case == 1


def plain_decorator(func):
    return func


@plain_decorator
def test_non_call_decorator():
    assert True


@pytest.mark.skip()
def test_unrelated_call_decorator():
    assert True


@pytest.mark.parametrize('operand')
def test_parametrize_without_values(operand):
    assert operand is not None


@pytest.mark.parametrize('operand', argvalues=unresolved_var)
def test_parametrize_unresolved_values(operand):
    assert operand is not None


@pytest.mark.parametrize('operand', argvalues=[])  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                             ^^
def test_empty_list_keyword(operand):
    assert double(operand) == operand * 2


@pytest.mark.parametrize('operand', argvalues=())  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                             ^^
def test_empty_tuple_keyword(operand):
    assert double(operand) == operand * 2


@pytest.mark.parametrize('operand', ([]))  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                    ^^
def test_empty_list_parenthesized(operand):
    assert double(operand) == operand * 2


not_a_list = None


@pytest.mark.parametrize('operand', not_a_list)  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^^^^^^^^^
def test_none_variable(operand):
    assert operand is None


aliased = []
other = aliased
other.append(1)


@pytest.mark.parametrize('operand', aliased)
def test_append_via_different_name(operand):
    assert operand == 1


split_alias_cases = []
left, right = split_alias_cases


@pytest.mark.parametrize('operand', split_alias_cases)  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^^^^^^^^^^^^^^^^
def test_unpacking_assignment_is_not_simple_alias(operand):
    assert operand is None


attr_alias_cases = []
holder = type('Holder', (), {})()
holder.cases = attr_alias_cases


@pytest.mark.parametrize('operand', attr_alias_cases)  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^^^^^^^^^^^^^^^
def test_assignment_to_attribute_is_not_alias_population(operand):
    assert operand is None


nested_population = []


def populate_nested_population():
    nested_population.append(1)


@pytest.mark.parametrize('operand', nested_population)  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^^^^^^^^^^^^^^^^
def test_append_only_inside_never_called_function(operand):
    assert operand == 1


@pytest.mark.parametrize('operand', set())  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^^^^
def test_empty_set_constructor(operand):
    assert operand is None


@pytest.mark.parametrize('operand', range(0))  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^^^^^^^
def test_empty_range_constructor(operand):
    assert operand is None


@pytest.mark.parametrize('operand', range(-1))  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^^^^^^^^
def test_empty_negative_range_constructor(operand):
    assert operand is None


@pytest.mark.parametrize('operand', range(1, 1))
def test_two_arg_empty_range_not_detected(operand):
    assert operand is None


@pytest.mark.parametrize('operand', range(0.5))
def test_float_range_argument_not_detected(operand):
    assert operand is None


n = 0


@pytest.mark.parametrize('operand', range(n))
def test_range_with_non_literal_argument(operand):
    assert operand is None


range_args = (0,)


@pytest.mark.parametrize('operand', range(*range_args))
def test_range_with_unpacked_argument(operand):
    assert operand is None


@pytest.mark.parametrize('operand', list())  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^^^^^
def test_empty_list_constructor(operand):
    assert double(operand) == operand * 2


@pytest.mark.parametrize('operand', dict())  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^^^^^
def test_empty_dict_constructor(operand):
    assert operand is None


@pytest.mark.parametrize('operand', tuple())  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^^^^^^
def test_empty_tuple_constructor(operand):
    assert operand is None


empty_set_var = set()


@pytest.mark.parametrize('operand', empty_set_var)  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^^^^^^^^^^^^
def test_empty_set_variable(operand):
    assert operand is None


populated_set = set()
populated_set.add(1)


@pytest.mark.parametrize('operand', populated_set)
def test_set_populated_via_add(operand):
    assert operand == 1


@pytest.mark.parametrize('operand', [])  # Noncompliant {{Add at least one case to the parametrize values.}}
#                                   ^^
def test_empty_literal_despite_unrelated_append(operand):
    assert operand is None


cases_with_attr_access = []
_ = cases_with_attr_access.append
cases_with_attr_access.append(1)


@pytest.mark.parametrize('operand', cases_with_attr_access)
def test_append_call_and_attribute_ref(operand):
    assert operand == 1
