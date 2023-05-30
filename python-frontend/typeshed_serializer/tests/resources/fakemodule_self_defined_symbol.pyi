from typing_extensions import TypeAlias

_ClassInfo: TypeAlias = type | tuple[_ClassInfo, ...]

class TestCase:
    def assertIsInstance(self, obj: object, cls: _ClassInfo) -> None: ...