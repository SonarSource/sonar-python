from SonarPythonAnalyzerFakeStub import CustomStubBase
from typing import List

from anthropic.types.content_block import ContentBlock

class Message(CustomStubBase):
    content: List[ContentBlock]
