from SonarPythonAnalyzerFakeStub import CustomStubBase

# Redis stubs exist in typeshed (typeshed_serializer/resources/typeshed/stubs/redis/redis/client.pyi),
# but they are not serialized properly (no StrictRedis stubs in the resulting protobuf for example).
# We choose to not serialize the typeshed stubs, but instead provide our own stubs here.
class Redis(CustomStubBase):
    def __init__(self, *args, **kwargs) -> None: ...

class StrictRedis(CustomStubBase):
    def __init__(self, *args, **kwargs) -> None: ...