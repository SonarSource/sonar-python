from SonarPythonAnalyzerFakeStub import CustomStubBase
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, Literal, Optional, Sequence
import httpx


class ClientV2(CustomStubBase):
    def __init__(
            self,
            api_key: Optional[Union[str, Callable[[], str]]] = None,
            *,
            base_url: Optional[str] = None,
            environment: ClientEnvironment = ...,
            client_name: Optional[str] = None,
            timeout: Optional[float] = None,
            httpx_client: Optional[httpx.Client] = None,
            thread_pool_executor: ThreadPoolExecutor = ...,
            log_warning_experimental_features: bool = True,
    ) -> None: ...

    def chat(
            self,
            *,
            model: str,
            messages: ChatMessages,
            tools: Optional[Sequence[ToolV2]] = Any,
            strict_tools: Optional[bool] = Any,
            documents: Optional[Sequence[V2ChatRequestDocumentsItem]] = Any,
            citation_options: Optional[CitationOptions] = Any,
            response_format: Optional[ResponseFormatV2] = Any,
            safety_mode: Optional[Literal["CONTEXTUAL", "STRICT", "OFF"]] = "CONTEXTUAL",
            max_tokens: Optional[int] = Any,
            stop_sequences: Optional[Sequence[str]] = Any,
            temperature: Optional[float] = Any,
            seed: Optional[int] = Any,
            frequency_penalty: Optional[float] = Any,
            presence_penalty: Optional[float] = Any,
            k: Optional[int] = Any,
            p: Optional[float] = Any,
            logprobs: Optional[bool] = Any,
            tool_choice: Optional[V2ChatRequestToolChoice] = Any,
            thinking: Optional[Thinking] = Any,
            request_options: Optional[RequestOptions] = None,
    ) -> V2ChatResponse: ...

    def chat_stream(
            self,
            *,
            model: str,
            messages: ChatMessages,
            tools: Optional[Sequence[ToolV2]] = Any,
            strict_tools: Optional[bool] = Any,
            documents: Optional[Sequence[V2ChatStreamRequestDocumentsItem]] = Any,
            citation_options: Optional[CitationOptions] = Any,
            response_format: Optional[ResponseFormatV2] = Any,
            safety_mode: Optional[Literal["CONTEXTUAL", "STRICT", "OFF"]] = "CONTEXTUAL",
            max_tokens: Optional[int] = Any,
            stop_sequences: Optional[Sequence[str]] = Any,
            temperature: Optional[float] = Any,
            seed: Optional[int] = Any,
            frequency_penalty: Optional[float] = Any,
            presence_penalty: Optional[float] = Any,
            k: Optional[int] = Any,
            p: Optional[float] = Any,
            logprobs: Optional[bool] = Any,
            tool_choice: Optional[V2ChatStreamRequestToolChoice] = Any,
            thinking: Optional[Thinking] = Any,
            request_options: Optional[RequestOptions] = None,
    ) -> Iterator[V2ChatStreamResponse]: ...
