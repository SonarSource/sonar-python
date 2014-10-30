from prod import factorial
def test_factorial_succeeding():
    assert factorial(2) == 2

def test_factorial_failing():
    assert factorial(3) == 2

def test_factorial_error():
    assert factorial(3) == 6
    raise RuntimeError()

