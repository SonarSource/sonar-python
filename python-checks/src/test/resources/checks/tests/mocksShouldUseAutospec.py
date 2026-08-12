import mock
import unittest.mock
from unittest.mock import MagicMock, Mock, create_autospec, patch


class PaymentGateway:
    def charge(self, amount):
        ...


class ApiClient:
    def fetch(self, id):
        ...


def checkout(payments, order):
    ...


def notify_user(user_id):
    ...


def test_bare_mock():
    payments = Mock()  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#              ^^^^
    checkout(payments, None)


def test_bare_magic_mock():
    client = MagicMock()  # Noncompliant {{Replace this "MagicMock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#            ^^^^^^^^^
    client.fetch("id")


def test_mock_with_return_value_only():
    # Discarded constructor — no collaborator interaction in the test.
    Mock(return_value=42)


def test_mock_with_side_effect_only():
    Mock(side_effect=ValueError)


def test_mock_with_name_only():
    Mock(name="payments")


def test_mock_with_wraps_only():
    Mock(wraps=PaymentGateway())


def test_fully_qualified_mock():
    unittest.mock.Mock()


def test_fully_qualified_magic_mock():
    unittest.mock.MagicMock()


def test_fully_qualified_mock_collaborator():
    payments = unittest.mock.Mock()  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#              ^^^^^^^^^^^^^^^^^^
    checkout(payments, None)


def test_third_party_mock():
    mock.Mock()


def test_third_party_magic_mock():
    mock.MagicMock()


def test_third_party_mock_collaborator():
    payments = mock.Mock()  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#              ^^^^^^^^^
    checkout(payments, None)


def test_patch_without_autospec():
    with patch("app.notify.send") as send:  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#        ^^^^^
        notify_user("u1")
        send.assert_called_once_with("u1")


def test_patch_as_decorator_unused_mock():
    # Isolation-only patch: mock parameter is never used.
    @patch("app.notify.send")
    def inner(send):
        notify_user("u1")

    inner()


def test_patch_as_decorator_used_mock():
    @patch("app.notify.send")  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#    ^^^^^
    def inner(send):
        notify_user("u1")
        send.assert_called_once_with("u1")

    inner()


def test_patch_object_without_autospec():
    with patch.object(PaymentGateway, "charge") as charge:  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#        ^^^^^^^^^^^^
        charge(10)


def test_fully_qualified_patch_unused():
    # No mock bound in the test — isolation only.
    with unittest.mock.patch("app.notify.send"):
        notify_user("u1")


def test_third_party_patch_unused():
    with mock.patch("app.notify.send"):
        notify_user("u1")


def test_mocker_patch_unused(mocker):
    mocker.patch("app.notify.send")


def test_mocker_patch_object_unused(mocker):
    mocker.patch.object(PaymentGateway, "charge")


def test_class_mocker_patch_unused(class_mocker):
    class_mocker.patch("app.notify.send")


def test_self_mocker_patch_unused():
    class Suite:
        def setup_method(self, mocker):
            self.mocker = mocker

        def test_it(self):
            self.mocker.patch("app.notify.send")


def test_mocker_patch_used(mocker):
    send = mocker.patch("app.notify.send")  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#          ^^^^^^^^^^^^
    notify_user("u1")
    send.assert_called_once_with("u1")


def test_patch_with_autospec_false_unused():
    # Isolation-only; autospec=False is irrelevant when the mock is unused.
    with patch("app.notify.send", autospec=False):
        notify_user("u1")


def test_patch_with_autospec_false_used():
    with patch("app.notify.send", autospec=False) as send:  # Noncompliant {{Replace "autospec=False" with "autospec=True", or pass an explicit "spec=" / "spec_set=".}}
#        ^^^^^
        notify_user("u1")
        send.assert_called_once_with("u1")


def test_field_stub_mock_not_raised():
    # Autospec cannot mirror instance fields populated in __init__.
    message = Mock()
    message.headers = {"id": "1"}
    message.body = b""


def test_field_stub_magic_mock_not_raised():
    response = MagicMock()
    response.status_code = 200
    response.raw = MagicMock()


def test_nested_mock_assigned_to_attribute():
    cur = Mock()
    connect = MagicMock()
    connect.return_value.cursor = cur
    cur.return_value.execute = Mock()
    cur.return_value.execute.assert_called()


def test_compliant_create_autospec():
    payments = create_autospec(PaymentGateway)
    checkout(payments, None)


def test_compliant_create_autospec_fully_qualified():
    client = unittest.mock.create_autospec(ApiClient)
    client.fetch("id")


def test_compliant_mock_with_spec():
    payments = Mock(spec=PaymentGateway)
    checkout(payments, None)


def test_compliant_mock_with_positional_spec():
    payments = Mock(PaymentGateway)
    checkout(payments, None)


def test_compliant_mock_with_spec_set():
    payments = Mock(spec_set=PaymentGateway)
    checkout(payments, None)


def test_compliant_magic_mock_with_spec():
    client = MagicMock(spec=ApiClient)
    client.fetch("id")


def test_compliant_magic_mock_with_spec_set():
    client = MagicMock(spec_set=ApiClient)
    client.fetch("id")


def test_compliant_patch_autospec_true():
    with patch("app.notify.send", autospec=True) as send:
        notify_user("u1")
        send.assert_called_once_with("u1")


def test_compliant_patch_autospec_class():
    with patch("app.notify.send", autospec=PaymentGateway):
        notify_user("u1")


def test_compliant_patch_with_spec():
    with patch("app.notify.send", spec=PaymentGateway):
        notify_user("u1")


def test_compliant_patch_with_spec_set():
    with patch("app.notify.send", spec_set=PaymentGateway):
        notify_user("u1")


def test_compliant_patch_object_autospec():
    with patch.object(PaymentGateway, "charge", autospec=True):
        ...


def test_compliant_patch_with_new():
    real = lambda user_id: None
    with patch("app.notify.send", new=real):
        notify_user("u1")


def test_compliant_patch_with_new_callable():
    with patch("app.notify.send", new_callable=lambda: lambda user_id: None):
        notify_user("u1")


def test_compliant_mocker_patch_autospec(mocker):
    mocker.patch("app.notify.send", autospec=True)


def test_compliant_mocker_patch_object_autospec(mocker):
    mocker.patch.object(PaymentGateway, "charge", autospec=True)


def test_compliant_decorator_autospec():
    @patch("app.notify.send", autospec=True)
    def inner(send):
        notify_user("u1")

    inner()


def test_ignored_when_unpacked_kwargs():
    kwargs = {"autospec": True}
    Mock(**kwargs)
    patch("app.notify.send", **kwargs)


def test_async_mock_not_covered():
    # RSpec only mentions Mock / MagicMock
    unittest.mock.AsyncMock()


def test_property_mock_not_covered():
    unittest.mock.PropertyMock()


def test_annotated_assignment_collaborator():
    payments: object = Mock()  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#                      ^^^^
    checkout(payments, None)


def test_chained_assignment_collaborator():
    payments = gateway = Mock()  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#                        ^^^^
    checkout(payments, None)


def test_inline_mock_argument():
    checkout(Mock(), None)  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#            ^^^^


def test_mock_called_directly():
    factory = Mock()  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#             ^^^^
    factory()


def test_fully_qualified_decorator_used():
    @unittest.mock.patch("app.notify.send")  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#    ^^^^^^^^^^^^^^^^^^^
    def inner(send):
        notify_user("u1")
        send.assert_called_once_with("u1")

    inner()


def test_patch_object_decorator_used():
    @patch.object(PaymentGateway, "charge")  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#    ^^^^^^^^^^^^
    def inner(charge):
        charge(10)

    inner()


@patch("app.notify.send")  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#^^^^^
class ClassLevelPatchSuite:
    def test_it(self, send):
        notify_user("u1")
        send.assert_called_once_with("u1")


def test_decorator_without_mock_parameter():
    # More patch decorators than injectable parameters — treat as unused.
    @patch("app.notify.send")
    def inner():
        notify_user("u1")

    inner()


def test_tuple_unpacking_mock_not_raised():
    # Unpacking yields child mocks, not the constructed mock itself.
    payments, order = Mock()
    checkout(payments, order)


def test_attribute_lhs_mock_not_raised_as_binding():
    gateway = PaymentGateway()
    gateway.charge = Mock()
    gateway.charge(10)


def test_patch_decorator_with_non_call_neighbor():
    # Non-call decorators must not break patch index → parameter mapping.
    @staticmethod
    @patch("app.notify.send")  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#    ^^^^^
    def inner(send):
        notify_user("u1")
        send.assert_called_once_with("u1")

    inner()


def test_parenthesized_mock_assignment():
    payments = (Mock())  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#               ^^^^
    checkout(payments, None)


def test_classmethod_patch_skips_cls_parameter():
    class Suite:
        @classmethod
        @patch("app.notify.send")  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#        ^^^^^
        def inner(cls, send):
            notify_user("u1")
            send.assert_called_once_with("u1")

    Suite.inner()


def test_instance_method_patch_skips_self_parameter():
    class Suite:
        @patch("app.notify.send")  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#        ^^^^^
        def inner(self, send):
            notify_user("u1")
            send.assert_called_once_with("u1")

    Suite().inner()


def test_with_patch_non_name_alias_ignored():
    # Alias is not a simple name — cannot track usages; treat as unused.
    class Box:
        pass

    box = Box()
    with patch("app.notify.send") as box.send:
        notify_user("u1")


def test_nested_mock_inside_mock_constructor():
    # Inner Mock is an argument to another Mock — not a collaborator use.
    Mock(Mock())


def test_patch_call_nested_inside_with_patch():
    # Inner patch is not the with-item test; isolation-only unless assigned/used.
    with patch("app.notify.send") as send:  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#        ^^^^^
        notify_user("u1")
        send.assert_called_once_with("u1")
        patch("app.notify.other")
