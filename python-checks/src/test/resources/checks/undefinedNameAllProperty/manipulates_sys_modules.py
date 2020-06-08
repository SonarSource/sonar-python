import sys
__all__ = ["something"] # OK


def foo():
  sys.module["something"] = get_something()
  foo["1"] = unknown["1"]
  bar()["2"] = foo()

