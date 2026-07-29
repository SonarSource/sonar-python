import warnings

import pytest
from pytest import warns as imported_warns


def emit_user_warning():
    warnings.warn("x", UserWarning)


def emit_deprecation_warning():
    warnings.warn("deprecated", DeprecationWarning)


class DummyContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


dummy_context_manager = DummyContextManager()


def test_warns_without_type():
    with pytest.warns():  # Noncompliant {{This assertion is too broad; use a more specific warning type or check the warning message.}}
#        ^^^^^^^^^^^^^^
        emit_user_warning()


def test_warns_broad_warning():
    with pytest.warns(Warning):  # Noncompliant {{This assertion is too broad; use a more specific warning type or check the warning message.}}
#                     ^^^^^^^
        emit_deprecation_warning()


def test_warns_expected_warning_keyword():
    with pytest.warns(expected_warning=Warning):  # Noncompliant {{This assertion is too broad; use a more specific warning type or check the warning message.}}
#                                      ^^^^^^^
        emit_deprecation_warning()


def test_imported_warns_broad():
    with imported_warns(Warning):  # Noncompliant {{This assertion is too broad; use a more specific warning type or check the warning message.}}
#                       ^^^^^^^
        emit_deprecation_warning()


def test_warns_empty_match():
    with pytest.warns(Warning, match=""):  # Noncompliant {{This assertion is too broad; use a more specific warning type or check the warning message.}}
#                     ^^^^^^^
        emit_deprecation_warning()


def test_warns_none_match():
    with pytest.warns(Warning, match=None):  # Noncompliant {{This assertion is too broad; use a more specific warning type or check the warning message.}}
#                     ^^^^^^^
        emit_deprecation_warning()


def test_direct_warns_without_type():
    pytest.warns()  # Noncompliant {{This assertion is too broad; use a more specific warning type or check the warning message.}}
#   ^^^^^^^^^^^^^^


def test_direct_warns_broad():
    pytest.warns(Warning, emit_deprecation_warning)  # Noncompliant {{This assertion is too broad; use a more specific warning type or check the warning message.}}
#                ^^^^^^^


def test_warns_tuple_containing_warning():
    with pytest.warns((Warning, UserWarning)):  # Noncompliant {{This assertion is too broad; use a more specific warning type or check the warning message.}}
#                      ^^^^^^^
        emit_user_warning()


def test_warns_list_containing_warning():
    with pytest.warns([Warning, DeprecationWarning]):  # Noncompliant {{This assertion is too broad; use a more specific warning type or check the warning message.}}
#                      ^^^^^^^
        emit_deprecation_warning()


def test_warns_nested_tuple_containing_warning():
    with pytest.warns(((Warning,), UserWarning)):  # Noncompliant {{This assertion is too broad; use a more specific warning type or check the warning message.}}
#                       ^^^^^^^
        emit_user_warning()


def test_warns_user_warning_without_match():
    with pytest.warns(UserWarning):
        emit_user_warning()


def test_warns_deprecation_warning_without_match():
    with pytest.warns(DeprecationWarning):
        emit_deprecation_warning()


def test_warns_specific_type():
    with pytest.warns(ResourceWarning):
        warnings.warn("resource", ResourceWarning)


def test_warns_tuple_without_warning():
    with pytest.warns((UserWarning, DeprecationWarning)):
        emit_user_warning()


def test_warns_tuple_with_warning_and_match():
    with pytest.warns((Warning, UserWarning), match="x"):
        emit_user_warning()


def test_warns_broad_with_match():
    with pytest.warns(Warning, match="deprecated"):
        emit_deprecation_warning()


def test_warns_match_only():
    with pytest.warns(match="x"):
        emit_user_warning()


def test_warns_match_only_variable():
    warning_message = "Can't encode more than one header value for the same key."
    with pytest.warns(match=warning_message):
        emit_user_warning()


def test_warns_match_only_empty():
    with pytest.warns(match=""):  # Noncompliant {{This assertion is too broad; use a more specific warning type or check the warning message.}}
#        ^^^^^^^^^^^^^^^^^^^^^^
        emit_user_warning()


def test_warns_match_only_none():
    with pytest.warns(match=None):  # Noncompliant {{This assertion is too broad; use a more specific warning type or check the warning message.}}
#        ^^^^^^^^^^^^^^^^^^^^^^^^
        emit_user_warning()


def test_warns_user_warning_with_match():
    with pytest.warns(UserWarning, match="x"):
        emit_user_warning()


def test_imported_warns_specific():
    with imported_warns(ResourceWarning):
        warnings.warn("resource", ResourceWarning)


def test_with_non_call_context_manager():
    with dummy_context_manager:
        emit_user_warning()


def test_unrelated_call():
    warnings.warn("x", UserWarning)
