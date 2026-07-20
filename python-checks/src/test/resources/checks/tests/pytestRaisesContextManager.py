import contextlib

import pytest
from pytest import raises as imported_raises


def process_data(data):
    return data.upper()


def bootstrap_session(df=10):
    if df <= 0:
        raise ValueError('df must be positive')
    if df > 20:
        raise NotImplementedError('df > 20 not supported')


def test_standalone_raises():
    pytest.raises(ValueError)  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
    process_data('hello')


def test_deprecated_callable_passing_form():
    pytest.raises(ValueError, bootstrap_session, df=-1)  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def test_typo_calling_function_immediately():
    pytest.raises(NotImplementedError, bootstrap_session(df=10))  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def test_compliant_with_block():
    with pytest.raises(ValueError):
        process_data(123)


def test_compliant_with_block_and_match():
    with pytest.raises(ValueError, match="must be positive"):
        bootstrap_session(df=-1)


def test_compliant_with_as_clause():
    with pytest.raises(ValueError) as exc_info:
        bootstrap_session(df=-1)
    assert "df must be positive" in str(exc_info.value)


def test_compliant_imported_raises():
    with imported_raises(ValueError):
        process_data(123)


def test_noncompliant_imported_raises():
    imported_raises(ValueError, bootstrap_session, df=-1)  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def test_non_pytest_call():
    process_data('hello')


def test_compliant_parenthesized_with_block():
    with (pytest.raises(ValueError)):
        process_data(123)


def test_compliant_saved_raises_reused_in_with():
    ctx = pytest.raises(ValueError)
    with ctx:
        process_data(123)


def test_compliant_saved_raises_reused_in_with_as():
    raises = pytest.raises(ValueError, match="must be positive")
    with raises as exc_info:
        bootstrap_session(df=-1)
    assert "df must be positive" in str(exc_info.value)


def test_compliant_saved_raises_conditional_then_with():
    if True:
        ctx = pytest.raises(TypeError, match="Invalid value")
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        process_data(123)


def test_compliant_saved_parenthesized_raises_reused_in_with():
    ctx = (pytest.raises(ValueError))
    with ctx:
        process_data(123)


def test_compliant_annotated_assignment_reused_in_with():
    ctx: object = pytest.raises(ValueError)
    with ctx:
        process_data(123)


def test_noncompliant_saved_raises_never_used_in_with():
    ctx = pytest.raises(ValueError)  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
#         ^^^^^^^^^^^^^^^^^^^^^^^^^
    process_data('hello')


def test_noncompliant_chained_assignment_even_if_used_in_with():
    # Chained assignment is not tracked as a single saved context manager.
    a = b = pytest.raises(ValueError)  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
#           ^^^^^^^^^^^^^^^^^^^^^^^^^
    with a:
        process_data(123)


def test_noncompliant_tuple_unpacking_assignment():
    ctx, other = pytest.raises(ValueError), None  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
#                ^^^^^^^^^^^^^^^^^^^^^^^^^
    process_data('hello')


def test_noncompliant_attribute_assignment():
    holder = type('Holder', (), {})()
    holder.ctx = pytest.raises(ValueError)  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
#                ^^^^^^^^^^^^^^^^^^^^^^^^^
    process_data('hello')


@pytest.mark.parametrize(
    "mode, expected_output, expectation",
    [
        pytest.param(
            None,
            "out",
            pytest.raises(ValueError),
            id="raises injected via pytest.param",
        ),
        pytest.param(
            None,
            "out",
            contextlib.nullcontext(),
            id="nullcontext",
        ),
    ],
)
def test_compliant_parametrize_pytest_param_raises_used_in_with(mode, expected_output, expectation):
    with expectation:
        process_data(123)


@pytest.mark.parametrize(
    "expectation",
    [pytest.raises(ValueError)],
)
def test_compliant_parametrize_direct_raises_used_in_with(expectation):
    with expectation:
        process_data(123)


@pytest.mark.parametrize(
    "x, expectation",
    [(1, pytest.raises(ValueError))],
)
def test_compliant_parametrize_tuple_raises_used_in_with(x, expectation):
    with expectation:
        process_data(123)


@pytest.mark.parametrize(
    ["expectation"],
    [pytest.raises(TypeError)],
)
def test_compliant_parametrize_list_argnames_raises_used_in_with(expectation):
    with expectation:
        process_data(123)


@pytest.mark.parametrize(
    "expectation",
    [pytest.raises(ValueError)],  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
#    ^^^^^^^^^^^^^^^^^^^^^^^^^
)
def test_noncompliant_parametrize_raises_never_used_in_with(expectation):
    process_data('hello')


@pytest.mark.parametrize(
    "x, expectation, unused",
    [
        pytest.param(
            1,
            contextlib.nullcontext(),
            pytest.raises(ValueError),  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
#           ^^^^^^^^^^^^^^^^^^^^^^^^^
        ),
    ],
)
def test_noncompliant_parametrize_raises_not_the_with_param(x, expectation, unused):
    with expectation:
        process_data(123)


@pytest.mark.parametrize(
    "expectation",
    [{pytest.raises(ValueError): None}],  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
#     ^^^^^^^^^^^^^^^^^^^^^^^^^
)
def test_noncompliant_parametrize_raises_in_unsupported_row_shape(expectation):
    with expectation:
        process_data(123)


@pytest.mark.parametrize(argvalues=[pytest.raises(ValueError)])  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
#                                   ^^^^^^^^^^^^^^^^^^^^^^^^^
def test_noncompliant_parametrize_missing_argnames(expectation):
    with expectation:
        process_data(123)


@pytest.mark.parametrize(*["expectation", [pytest.raises(TypeError)]])  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
#                                          ^^^^^^^^^^^^^^^^^^^^^^^^
def test_noncompliant_parametrize_starred_args(expectation):
    with expectation:
        process_data(123)


@pytest.mark.parametrize(
    "expectation,",
    [pytest.raises(ValueError)],
)
def test_compliant_parametrize_argnames_trailing_comma(expectation):
    with expectation:
        process_data(123)


def test_noncompliant_annotated_attribute_assignment():
    holder = type('Holder', (), {})()
    holder.ctx: object = pytest.raises(ValueError)  # Noncompliant {{Prefer the context manager form: wrap the raising code in "with pytest.raises(...)".}}
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^
    process_data('hello')


def external_error_raised(expected_exception):
    # Escapes via return for callers that use: with external_error_raised(...):
    return pytest.raises(expected_exception, match=None)


def external_error_raised_parenthesized(expected_exception):
    return (pytest.raises(expected_exception, match=None))


def external_error_raised_via_local(expected_exception):
    ctx = pytest.raises(expected_exception, match=None)
    return ctx


_MODULE_RAISES = pytest.raises(ValueError)


def assign_raises_to_global():
    global _MODULE_RAISES
    _MODULE_RAISES = pytest.raises(TypeError)


class RaisesHolder:
    def store_raises_on_self(self):
        self.ctx = pytest.raises(ValueError)

    def store_raises_on_self_annotated(self):
        self.ctx: object = pytest.raises(ValueError)


def escapes_via_argument_to_other_call(expected_exception):
    # Escapes as a call argument — caller may use it in with.
    return process_data(pytest.raises(expected_exception, match=None))


def escapes_via_direct_call_argument(expected_exception):
    process_data(pytest.raises(expected_exception, match=None))


def escapes_via_local_passed_as_argument(expected_exception):
    ctx = pytest.raises(expected_exception, match=None)
    process_data(ctx)
