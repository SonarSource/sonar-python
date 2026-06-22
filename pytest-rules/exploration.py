import pytest

### S8714
def test_non_compliant_s8714():
    try:
        1 / 0
        pytest.fail("Compliant S8714")
    except ZeroDivisionError:
        pass

def test_compliant_s8714():
    with pytest.raises(ZeroDivisionError):
        1 / 0

### S5958
def test_non_compliant_s5958() :
    with pytest.raises():
        1 / 0

def test_compliant_s5958() :
    with pytest.raises(ZeroDivisionError):
        1 / 0
    # or
    with pytest.raises(Exception, match="division by zero"):
        1 / 0

# new rule ideas:
# - dont use xfail, fail, ... in non-test functions
#    https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest-xfail
#    This function should be called only during testing (setup, call or teardown).