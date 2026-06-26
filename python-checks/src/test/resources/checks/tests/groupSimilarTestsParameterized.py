def setup_tax():
    pass


def get_tax(value):
    return value


def test_not_null1():
    setup_tax()
    assert get_tax(1) is not None


def test_not_null2():
    setup_tax()
    assert get_tax(2) is not None


def test_not_null3():
    setup_tax()
    assert get_tax(3) is not None
