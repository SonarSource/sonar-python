__all__ = ["my_totally_unknown_name"]

class A: ...

def __getattr__(name):
  if name == "my_totally_unknown_name":
    return A
  else:
    raise AttributeError()

def foo(): ...

def __dir__(name): ...
