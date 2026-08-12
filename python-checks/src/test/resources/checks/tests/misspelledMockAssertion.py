from unittest.mock import AsyncMock, MagicMock, Mock, patch


def process(saver, item):
    saver.save(item)


def notify_user(user_id):
    pass


def test_assert_called_with_typo():
    saver = Mock()
    process(saver, "item")
    saver.save.assert_called_wit("item")  # Noncompliant {{Correct this misspelled mock assertion; did you mean "assert_called_with"?}}
#              ^^^^^^^^^^^^^^^^^


def test_assert_called_once_with_typo():
    mock = Mock()
    mock(1)
    mock.assert_caled_once_with(1)  # Noncompliant {{Correct this misspelled mock assertion; did you mean "assert_called_once_with"?}}
#        ^^^^^^^^^^^^^^^^^^^^^^


def test_assert_called_typo_on_magic_mock():
    mock = MagicMock()
    mock()
    mock.assert_calld()  # Noncompliant {{Correct this misspelled mock assertion; did you mean "assert_called"?}}
#        ^^^^^^^^^^^^


def test_assert_not_called_typo():
    mock = Mock()
    mock.assert_not_calld()  # Noncompliant {{Correct this misspelled mock assertion; did you mean "assert_not_called"?}}
#        ^^^^^^^^^^^^^^^^


def test_assert_any_call_typo():
    mock = Mock()
    mock(1)
    mock.assert_any_cal(1)  # Noncompliant {{Correct this misspelled mock assertion; did you mean "assert_any_call"?}}
#        ^^^^^^^^^^^^^^


def test_assert_has_calls_typo():
    mock = Mock()
    mock.assert_has_call([])  # Noncompliant {{Correct this misspelled mock assertion; did you mean "assert_has_calls"?}}
#        ^^^^^^^^^^^^^^^


def test_async_mock_await_typo():
    mock = AsyncMock()
    mock.assert_awaited_wit()  # Noncompliant {{Correct this misspelled mock assertion; did you mean "assert_awaited_with"?}}
#        ^^^^^^^^^^^^^^^^^^


def test_patch_context_manager_typo():
    with patch("os.path.exists") as exists:
        exists.assert_called_once_wit()  # FN - patch return type is unknown


def test_correct_assertions_are_ok():
    mock = Mock()
    mock(1, 2)
    mock.assert_called()
    mock.assert_called_once()
    mock.assert_called_with(1, 2)
    mock.assert_called_once_with(1, 2)
    mock.assert_any_call(1, 2)
    mock.assert_has_calls([])
    mock.assert_not_called()


def test_async_correct_assertions_are_ok():
    mock = AsyncMock()
    mock.assert_awaited()
    mock.assert_awaited_once()
    mock.assert_awaited_with()
    mock.assert_awaited_once_with()
    mock.assert_any_await()
    mock.assert_has_awaits([])
    mock.assert_not_awaited()


def test_unrelated_attributes_are_ok():
    mock = Mock()
    mock.return_value = 1
    mock.side_effect = None
    mock.reset_mock()
    mock.configure_mock(foo=1)
    # Far from any assertion name: not raised (safe).
    mock.completely_different_name()
    # Looks a bit like assert but too distant from known methods.
    mock.assert_something_totally_custom()


def test_non_mock_objects_are_ok():
    class Helper:
        def assert_called_wit(self, *_args):
            pass

    Helper().assert_called_wit(1)


def test_exact_custom_member_matching_known_name_is_ok():
    # Exact known assertion names must never be flagged.
    mock = Mock()
    mock.assert_called_with(1)
