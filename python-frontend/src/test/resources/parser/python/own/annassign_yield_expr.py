def _bar(current):
    return current

def _foo():
    x: int = 0
    y: int = yield _bar(x)
    y: int = yield a, a
    yield x + (y if y else 0)
