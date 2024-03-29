from SonarPythonAnalyzerFakeStub import CustomStubBase

class Message(CustomStubBase): ...

class Mail(CustomStubBase):
    def connect(self, *args, **kwargs) -> None: ...
    def send(self, *args, **kwargs) -> None: ...
    def send_message(self, *args, **kwargs) -> None: ...

class Connection(CustomStubBase):
    def send(self, *args, **kwargs) -> None: ...
    def send_message(self, *args, **kwargs) -> None: ...
