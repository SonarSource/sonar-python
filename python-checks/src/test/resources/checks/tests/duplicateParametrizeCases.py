import pytest


# Duplicate int literals
@pytest.mark.parametrize("n", [
    1,
    2,
    2,  # Noncompliant {{Remove this duplicate test case.}}
#   ^
#   ^@-2< {{Original.}}
    3,
])
def test_duplicate_ints(n):
    return n * 2


# Duplicate string literals
@pytest.mark.parametrize("value", [
    "a",
    "b",
    "a",  # Noncompliant {{Remove this duplicate test case.}}
#   ^^^
])
def test_duplicate_strings(value):
    assert value in ("a", "b")


# Duplicate tuple literals
@pytest.mark.parametrize("point", [
    (1, 2),
    (3, 4),
    (1, 2),  # Noncompliant {{Remove this duplicate test case.}}
#   ^^^^^^
])
def test_duplicate_tuples(point):
    operand, expected = point
    assert operand < expected


# Duplicate names
@pytest.mark.parametrize("case", [
    FIRST,
    SECOND,
    FIRST,  # Noncompliant {{Remove this duplicate test case.}}
#   ^^^^^
])
def test_duplicate_names(case):
    registry.append(case)


# Duplicate call expressions
@pytest.mark.parametrize("case", [
    make_case(1),
    make_case(2),
    make_case(1),  # Noncompliant
])
def test_duplicate_calls(case):
    consume(case)


# Three identical cases: both the second and third are flagged against the first
@pytest.mark.parametrize("n", [
    7,
    7,  # Noncompliant
    7,  # Noncompliant
])
def test_triple_duplicate(n):
    counter.increment(n)


# Two independent duplicate groups produce two separate issues
@pytest.mark.parametrize("case", [
    A,
    B,
    A,  # Noncompliant
    B,  # Noncompliant
])
def test_two_duplicate_groups(case):
    tracker.record(case)


# Duplicate detected when argvalues is passed as a keyword argument
@pytest.mark.parametrize("n", argvalues=[
    8,
    8,  # Noncompliant
])
def test_duplicate_via_keyword_argument(n):
    pipeline.run(n)


# Duplicate detected when argvalues is a tuple literal rather than a list
@pytest.mark.parametrize("n", (
    9,
    9,  # Noncompliant
))
def test_duplicate_in_tuple_argvalues(n):
    logger.debug(n)


# Parenthesized list is still a list literal once parentheses are stripped
@pytest.mark.parametrize("n", ([
    10,
    10,  # Noncompliant
]))
def test_duplicate_in_parenthesized_list(n):
    yield n


# Parentheses around a case must not hide equivalence with a bare literal
@pytest.mark.parametrize("n", [
    1,
    (1),  # Noncompliant {{Remove this duplicate test case.}}
#   ^^^
#   ^@-2< {{Original.}}
])
def test_parenthesized_case_duplicate(n):
    assert n == 1


# Nested parentheses around a case
@pytest.mark.parametrize("n", [
    ((2)),
    2,  # Noncompliant {{Remove this duplicate test case.}}
#   ^
])
def test_nested_parenthesized_case_duplicate(n):
    assert n == 2


# Compliant: all cases are distinct
@pytest.mark.parametrize("n", [1, 2, 3])
def test_distinct_ints(n):
    history.append(n)


# Compliant: distinct tuples
@pytest.mark.parametrize("operand,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_distinct_tuples(operand, expected):
    assert operand * 2 == expected


# Compliant: single-element list can never contain a duplicate
@pytest.mark.parametrize("n", [1])
def test_single_case(n):
    ...


# Compliant: values that only look similar are not equivalent expressions
@pytest.mark.parametrize("value", [1, 1.0, True])
def test_distinct_literal_kinds(value):
    kinds.add(type(value))


# Compliant: non-literal argvalues (a variable) is not inspected, even if it holds duplicates
cases_with_duplicates = [1, 2, 2, 3]


@pytest.mark.parametrize("n", cases_with_duplicates)
def test_non_literal_argvalues_skipped(n):
    seen.add(n)


# Compliant: argvalues produced by a call is not inspected
def build_cases():
    return [1, 2, 2, 3]


@pytest.mark.parametrize("n", build_cases())
def test_call_argvalues_skipped(n):
    processed.add(n)


# Compliant: no argvalues provided at all
@pytest.mark.parametrize("n")
def test_parametrize_without_values(n):
    assert n is not None


# Compliant: decorator that is not a call expression
def plain_decorator(func):
    return func


@plain_decorator
def test_non_call_decorator():
    return True


# Compliant: unrelated pytest.mark call
@pytest.mark.skip()
def test_unrelated_call_decorator():
    pass
