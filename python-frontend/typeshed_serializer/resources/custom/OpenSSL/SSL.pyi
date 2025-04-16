from SonarPythonAnalyzerFakeStub import CustomStubBase

from typing import Any

VERIFY_NONE: int

class Context(CustomStubBase):
    def set_verify(self, *args, **kwargs) -> None: ...
    def set_cipher_list(self, *args, **kwargs) -> None: ...

class Connection(CustomStubBase):
    ...
