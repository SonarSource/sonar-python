import mock
import unittest.mock
from unittest.mock import patch


class Target:
    attribute = None


def helper(*args, **kwargs):
    ...


def test_patch_positional_lambda():
    with patch("app.api.fetch", lambda: {"ok": True}):  # Noncompliant {{Replace this lambda with a "return_value" argument.}}
#                               ^^^^^^^^^^^^^^^^^^^^
        helper()


def test_patch_keyword_lambda():
    with patch("app.api.fetch", new=lambda: 42):  # Noncompliant
#                                   ^^^^^^^^^^
        helper()


def test_patch_lambda_with_unused_star_args():
    patch("app.api.fetch", lambda *args: 42)  # Noncompliant
#                          ^^^^^^^^^^^^^^^^


def test_patch_lambda_with_unused_kwargs():
    patch("app.api.fetch", lambda **kwargs: 42)  # Noncompliant
#                          ^^^^^^^^^^^^^^^^^^^


def test_patch_lambda_with_unused_parameters():
    patch("app.api.fetch", lambda x, y: 7)  # Noncompliant
#                          ^^^^^^^^^^^^^^


def test_patch_lambda_with_unused_default_parameter():
    patch("app.api.fetch", lambda x=1: 7)  # Noncompliant
#                          ^^^^^^^^^^^^^


def test_patch_lambda_in_parentheses():
    patch("app.api.fetch", (lambda: 42))  # Noncompliant
#                           ^^^^^^^^^^


def test_patch_object_positional_lambda():
    patch.object(Target, "attribute", lambda: 42)  # Noncompliant
#                                     ^^^^^^^^^^


def test_patch_object_keyword_lambda():
    patch.object(Target, "attribute", new=lambda: 42)  # Noncompliant
#                                         ^^^^^^^^^^


def test_fully_qualified_patch():
    unittest.mock.patch("app.api.fetch", lambda: 42)  # Noncompliant
#                                        ^^^^^^^^^^


def test_third_party_mock_patch():
    mock.patch("app.api.fetch", lambda: 42)  # Noncompliant
#                               ^^^^^^^^^^


@patch("app.api.fetch", lambda: 42)  # Noncompliant
#                       ^^^^^^^^^^
def test_patch_as_decorator(mocked):
    helper()


@patch("app.api.fetch", lambda *args: (helper(), None))
def test_compliant_decorator_lambda_with_call(mocked):
    # return_value= would evaluate the call at decoration time, not on each patch invocation
    helper()


@patch.object(Target, "attribute", lambda: helper())
def test_compliant_decorator_object_lambda_with_call(mocked):
    helper()


def test_patch_lambda_with_call_not_decorator():
    # Non-decorator patches keep the issue even when the body has a call (no quick fix offered)
    with patch("app.api.fetch", lambda *args: (helper(), None)):  # Noncompliant
#                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        helper()


def test_self_mocker_patch():
    class TestX:
        def setup_method(self, method, mocker):
            self.mocker = mocker

        def test_x(self):
            self.mocker.patch("module.target", lambda: 7)  # Noncompliant
#                                              ^^^^^^^^^


def test_self_mocker_patch_object():
    class TestX:
        def setup_method(self, method, mocker):
            self.mocker = mocker

        def test_x(self):
            self.mocker.patch.object(Target, "attribute", lambda: 7)  # Noncompliant
#                                                         ^^^^^^^^^


def test_mocker_patch(mocker):
    mocker.patch("module.target", lambda: 7)  # Noncompliant
#                                 ^^^^^^^^^


def test_mocker_patch_object(mocker):
    mocker.patch.object(Target, "attribute", lambda: 7)  # Noncompliant
#                                            ^^^^^^^^^


def test_session_mocker_patch(session_mocker):
    session_mocker.patch("module.target", lambda: 7)  # Noncompliant
#                                         ^^^^^^^^^


def test_module_mocker_patch(module_mocker):
    module_mocker.patch("module.target", lambda: 7)  # Noncompliant
#                                        ^^^^^^^^^


def test_class_mocker_patch(class_mocker):
    class_mocker.patch("module.target", lambda: 7)  # Noncompliant
#                                       ^^^^^^^^^


def test_package_mocker_patch(package_mocker):
    package_mocker.patch("module.target", lambda: 7)  # Noncompliant
#                                         ^^^^^^^^^


def test_lambda_returning_outer_name(mocker):
    sentinel = object()
    mocker.patch("module.target", lambda: sentinel)  # Noncompliant
#                                 ^^^^^^^^^^^^^^^^


def test_lambda_body_ignoring_its_parameter(mocker):
    mocker.patch("module.target", lambda value: helper())  # Noncompliant
#                                 ^^^^^^^^^^^^^^^^^^^^^^


def test_lambda_returning_enclosing_parameter(mocker, expected):
    mocker.patch("module.target", lambda value: expected)  # Noncompliant
#                                 ^^^^^^^^^^^^^^^^^^^^^^


def test_patch_lambda_with_unused_keyword_only_parameter():
    patch("app.api.fetch", lambda *, value: 7)  # Noncompliant
#                          ^^^^^^^^^^^^^^^^^^


def test_multiline_lambda_body():
    patch("app.api.fetch", lambda: {  # Noncompliant
#                          ^[el=+3;ec=5]
        "ok": True,
    })


def test_compliant_return_value():
    with patch("app.api.fetch", return_value={"ok": True}):
        helper()


def test_compliant_lambda_using_parameters():
    patch("app.api.fetch", lambda x, y: x + y)


def test_compliant_lambda_using_star_args():
    patch("app.api.fetch", lambda *args: args[0])


def test_compliant_lambda_using_kwargs():
    patch("app.api.fetch", lambda **kwargs: kwargs["value"])


def test_compliant_lambda_using_parameter_in_nested_call(mocker):
    mocker.patch("module.target", lambda value: helper(value))


def test_compliant_identity_lambda():
    patch("app.api.fetch", lambda value: value)


def test_compliant_lambda_using_keyword_only_parameter():
    patch("app.api.fetch", lambda *, value: value)


def test_compliant_side_effect_lambda():
    patch("app.api.fetch", side_effect=lambda: 42)


def test_compliant_patch_with_named_function():
    patch("app.api.fetch", helper)


def test_compliant_patch_dict():
    patch.dict("app.api.settings", {"a": 1})


def test_compliant_patch_multiple():
    patch.multiple("app.api", fetch=lambda: 42)


def test_compliant_patch_with_unpacked_arguments(arguments):
    patch("app.api.fetch", *arguments)


def test_compliant_mocker_patch_return_value(mocker):
    mocker.patch("module.target", return_value=7)


def test_compliant_unrelated_receiver(unknown_object):
    unknown_object.patch("module.target", lambda: 7)


def test_compliant_patch_on_call_result():
    helper().patch("module.target", lambda: 7)


def test_compliant_object_on_unrelated_receiver(unknown_object):
    unknown_object.object(Target, "attribute", lambda: 7)


def test_compliant_object_on_unrelated_patch_owner():
    helper.patch.object(Target, "attribute", lambda: 7)


def test_compliant_object_on_unrelated_member():
    helper.other.object(Target, "attribute", lambda: 7)


def test_compliant_locally_assigned_mocker():
    mocker = helper
    mocker.patch("module.target", lambda: 7)


def test_compliant_mocker_aliased_to_local(mocker):
    # Accepted FN: local aliases of the mocker fixture are not tracked
    m = mocker
    m.patch("module.target", lambda: 7)


def local_patch(target, new):
    ...


def test_compliant_locally_defined_patch():
    local_patch("app.api.fetch", lambda: 42)


def test_compliant_lambda_as_first_argument():
    helper(lambda: 42)
