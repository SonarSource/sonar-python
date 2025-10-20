from langchain_core.outputs.generation import Generation

from SonarPythonAnalyzerFakeStub import CustomStubBase


class LLMResult(CustomStubBase):
    generations: list[list[Generation]]
