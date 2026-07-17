from SonarPythonAnalyzerFakeStub import CustomStubBase

class FixtureRequest(CustomStubBase):
    def getfixturevalue(self, argname: str) -> object: ...
