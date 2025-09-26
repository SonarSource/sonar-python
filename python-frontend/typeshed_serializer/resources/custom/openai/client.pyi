from SonarPythonAnalyzerFakeStub import CustomStubBase
from openai.resources.chat.completions.completions import Completions
from openai.resources.chat import Chat
from openai.resources.beta import Beta
from openai.resources.responses import Responses

class OpenAI(CustomStubBase):
    chat: Chat
    completions: Completions
    beta: Beta
    responses: Responses
