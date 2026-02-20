from typing import Optional, List
from portkey_ai.api_resources.types.chat_complete_type import Choice, Usage, StreamChoice
from pydantic import BaseModel

from SonarPythonAnalyzerFakeStub import CustomStubBase


class ChatCompletions(BaseModel, CustomStubBase):
    id: Optional[str]
    choices: Optional[List[Choice]]
    created: Optional[int]
    model: Optional[str]
    object: Optional[str]
    system_fingerprint: Optional[str]
    usage: Optional[Usage]
    service_tier: Optional[str]

class ChatCompletionChunk(BaseModel, CustomStubBase):
    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choices: Optional[List[StreamChoice]] = None
    service_tier: Optional[str] = None
    system_fingerprint: Optional[str] = None
    usage: Optional[Usage] = None
