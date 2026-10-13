def add_protocol(cls):
    cls.__add__ = lambda self, other: 42
    cls.__class_getitem__ = classmethod(lambda cls, index: 42)
    cls.__getitem__ = lambda self, index: 42
    cls.__setitem__ = lambda self, index, value: None
    cls.__delitem__ = lambda self, index: None
    cls.__neg__ = lambda self: 42
    cls.__pos__ = lambda self: 42
    cls.__invert__ = lambda self: 42
    return cls


@add_protocol
class ImportedBase:
    pass
