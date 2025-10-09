from SonarPythonAnalyzerFakeStub import CustomStubBase
from typing import List

from anthropic.types.beta.beta_content_block import BetaContentBlock


class BetaMessage(CustomStubBase):
    content: List[BetaContentBlock]
    type: str = "message"
