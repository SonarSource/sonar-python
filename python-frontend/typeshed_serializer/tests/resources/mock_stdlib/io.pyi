import abc
class IOBase(metaclass=abc.ABCMeta):
    ...

from abc import ABCMeta
class NameMeta(metaclass=ABCMeta):
    ...

class StrMeta(metaclass="abc"):
    ...