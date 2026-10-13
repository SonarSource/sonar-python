# Project-local module shadowing typing: this function changes the class at runtime.
def final(cls):
    cls.__add__ = lambda self, other: 42
    cls.__class_getitem__ = classmethod(lambda cls, key: 42)
    return cls
