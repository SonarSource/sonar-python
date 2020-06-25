from typing import Tuple
from .channel import Channel

class SSHClient(CustomStubBase):
    def exec_command(self, *args, **kwargs) -> Tuple: ...
    def invoke_shell(self, *args, **kwargs) -> Channel: ...

