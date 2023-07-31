from abc import ABC, abstractmethod
import abc

from some_module import some_decorator

class Test(ABC):
    @abstractmethod
    def standard_usage(self):
        pass

    @abc.abstractmethod
    def qualified_usage(self):
        pass

    @some_decorator
    @abc.abstractmethod
    def usage_with_other_decorator(self):
        pass

    @abstractmethod()
    def incorrect_calling_usage(self):
        pass

    @unknown_decorator_symbol
    @abstractmethod
    def usage_with_unknown_other_decorator(self):
        pass

    def standard_method(self):
        pass

    @some_decorator
    def with_other_decorator(self):
        pass

    @unknown_decorator_symbol
    def with_unknown_decorator(self):
        pass
