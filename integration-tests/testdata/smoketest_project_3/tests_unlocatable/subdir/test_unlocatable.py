# these cannot be located because the __init__ file is missing
# (and thus the report doesnt contain full paths)
def test_succeeding():
    assert True

def test_failing():
    assert False

def test_crashing():
    raise RuntimeError()

def test_skipped():
    from nose.plugins.skip import SkipTest
    raise SkipTest()
