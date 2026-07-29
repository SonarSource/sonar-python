from pytest import fail as imported_fail


def test_assert_false_with_from_import():
    assert False  # Noncompliant {{Replace this assertion with pytest.fail(...) and provide a message.}}
#   ^^^^^^^^^^^^
    imported_fail("ready")
