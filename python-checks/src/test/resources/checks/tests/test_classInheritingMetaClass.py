import unittest


def test_module_level():
    pass


class TestMeta(type):
    pass


class SharedWithMetaClass(unittest.TestCase, metaclass=TestMeta):
    pass


class SharedWithLegacyMetaClass(unittest.TestCase):
    __metaclass__ = TestMeta
    pass


class TestInheritingMetaClass(SharedWithMetaClass):
    def helper(self):
        return 42
