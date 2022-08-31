from pytest import raises as my_raise

def test_assert_not_unittest():
    with my_raise(ZeroDivisionError):
        assert bar() == 42  # Noncompliant
