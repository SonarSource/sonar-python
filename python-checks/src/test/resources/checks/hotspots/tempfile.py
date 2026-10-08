def TemporaryFile(dir):
    pass


class NamedTemporaryFile:
    def __init__(self, dir):
        pass


class TemporaryDirectory:
    def __init__(self, dir):
        pass


from typing import overload

@overload
def mkstemp(dir: str): ...

@overload
def mkstemp(dir: bytes): ...

def mkstemp(dir):
    pass

if condition:
    def mkdtemp(dir):
        pass
else:
    def mkdtemp(dir):
        pass
