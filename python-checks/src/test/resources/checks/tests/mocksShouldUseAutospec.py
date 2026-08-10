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
    Mock(return_value=42)  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#   ^^^^


def test_mock_with_side_effect_only():
    Mock(side_effect=ValueError)  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#   ^^^^


def test_mock_with_name_only():
    Mock(name="payments")  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#   ^^^^


def test_mock_with_wraps_only():
    Mock(wraps=PaymentGateway())  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#   ^^^^


def test_fully_qualified_mock():
    unittest.mock.Mock()  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#   ^^^^^^^^^^^^^^^^^^


def test_fully_qualified_magic_mock():
    unittest.mock.MagicMock()  # Noncompliant {{Replace this "MagicMock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#   ^^^^^^^^^^^^^^^^^^^^^^^


def test_third_party_mock():
    mock.Mock()  # Noncompliant {{Replace this "Mock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#   ^^^^^^^^^


def test_third_party_magic_mock():
    mock.MagicMock()  # Noncompliant {{Replace this "MagicMock()" with "create_autospec(<collaborator>)", or pass "spec=" / "spec_set=".}}
#   ^^^^^^^^^^^^^^


def test_patch_without_autospec():
    with patch("app.notify.send") as send:  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#        ^^^^^
        notify_user("u1")
        send.assert_called_once_with("u1")


def test_patch_as_decorator():
    @patch("app.notify.send")  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#    ^^^^^
    def inner(send):
        notify_user("u1")

    inner()


def test_patch_object_without_autospec():
    with patch.object(PaymentGateway, "charge") as charge:  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#        ^^^^^^^^^^^^
        charge(10)


def test_fully_qualified_patch():
    with unittest.mock.patch("app.notify.send"):  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#        ^^^^^^^^^^^^^^^^^^^
        notify_user("u1")


def test_third_party_patch():
    with mock.patch("app.notify.send"):  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#        ^^^^^^^^^^
        notify_user("u1")


def test_mocker_patch(mocker):
    mocker.patch("app.notify.send")  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#   ^^^^^^^^^^^^


def test_mocker_patch_object(mocker):
    mocker.patch.object(PaymentGateway, "charge")  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#   ^^^^^^^^^^^^^^^^^^^


def test_class_mocker_patch(class_mocker):
    class_mocker.patch("app.notify.send")  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#   ^^^^^^^^^^^^^^^^^^


def test_self_mocker_patch():
    class Suite:
        def setup_method(self, mocker):
            self.mocker = mocker

        def test_it(self):
            self.mocker.patch("app.notify.send")  # Noncompliant {{Add "autospec=True" to this patch call, or pass an explicit "spec=" / "spec_set=".}}
#           ^^^^^^^^^^^^^^^^^


def test_patch_with_autospec_false():
    with patch("app.notify.send", autospec=False):  # Noncompliant {{Replace "autospec=False" with "autospec=True", or pass an explicit "spec=" / "spec_set=".}}
#        ^^^^^
        notify_user("u1")


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
