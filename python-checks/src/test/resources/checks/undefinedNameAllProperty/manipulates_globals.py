__all__ = ["another_yet_unknown_name"]


def foo(): ...


def bar():
    globals().update({"another_yet_unknown_name": foo})

bar()
