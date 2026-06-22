import pytest


### S8714 : Dedicated exception assertions should be used instead of "try-catch" with "fail()"
def test_non_compliant_s8714():
    try:
        1 / 0
        pytest.fail("Compliant S8714")
    except ZeroDivisionError:
        pass


def test_compliant_s8714():
    with pytest.raises(ZeroDivisionError):
        1 / 0


# S5958 : AssertJ "assertThatThrownBy" should not be used alone
#      -> pytest.raises should not be used for generic exceptions without message
def test_non_compliant_s5958():
    with pytest.raises(Exception):
        1 / 0


def test_compliant_s5958():
    with pytest.raises(ZeroDivisionError):
        1 / 0
    # or
    with pytest.raises(Exception, match="division by zero"):
        1 / 0


# S5976 : Similar tests should be grouped in a single Parameterized test
def test_non_compliant_s5976_1():
    with pytest.raises(ZeroDivisionError):
        1 / 0


def test_non_compliant_s5976_2():
    with pytest.raises(ZeroDivisionError):
        2 / 0


@pytest.mark.parametrize("numerator", [1, 2])
def test_compliant_s5976(numerator):
    with pytest.raises(ZeroDivisionError):
        _ = numerator / 0

# S2699 : Tests should include assertions
def test_non_compliant_s2699():
    pass

def test_compliant_s2699_0():
    assert 0 == 0

def test_compliant_s2699_1():
    if False :
        pytest.fail("Compliant S2699")

def test_compliant_s2699_2():
    with pytest.raises(ZeroDivisionError):
        1 / 0

# S5778: Only one method invocation is expected when testing runtime exceptions
def compute_something(x):
    return x / 0

def do_something(x):
    raise RuntimeError("generic failure")

def test_non_compliant_s5778():
    with pytest.raises(ZeroDivisionError):
        # we don't know which one raised
        do_something(compute_something(1))

def test_compliant_s5778():
    with pytest.raises(ZeroDivisionError):
        compute_something(1)
    with pytest.raises(RuntimeError, match="generic failure"):
        do_something(1)


# new rule ideas:
# - dont use xfail, fail, ... in non-test functions
#    https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest-xfail
#    This function should be called only during testing (setup, call or teardown).
# - use parametrized tests instead of tests with single for loop at top level
#    pytest.mark.parametrize : https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest-mark-parametrize
