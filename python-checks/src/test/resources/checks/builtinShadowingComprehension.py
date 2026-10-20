def f():
    int = 1
    values = [(int := x) for x in range(2)]
    return int
