# No pytest import: assert False / assert 0 must not be flagged
# (cannot know whether the runner is pytest or unittest).


def test_assert_false_without_pytest():
    assert False


def test_assert_zero_without_pytest():
    assert 0


def test_assert_false_with_message_without_pytest():
    assert False, "not ready yet"


class TestWithoutPytest:
    def test_assert_false(self):
        assert False

    def test_assert_zero(self):
        assert 0


def fail():
    pass


def test_local_fail_without_pytest():
    fail()
