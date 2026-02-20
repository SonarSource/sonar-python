from portkey_ai.api_resources.types.chat_complete_type import ChatCompletions, ChatCompletionChunk
from SonarPythonAnalyzerFakeStub import CustomStubBase
from portkey_ai.api_resources.apis.api_resource import APIResource, AsyncAPIResource
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Optional, Union
from portkey_ai._vendor.openai._types import Omit, omit


class Completions(APIResource, CustomStubBase):
    def create(
        self,
        *,
        model: Optional[str] = "portkey-default",
        messages: Iterable[Any],
        stream: Union[bool, Omit] = omit,
        temperature: Union[float, Omit] = omit,
        max_tokens: Union[int, Omit] = omit,
        top_p: Union[float, Omit] = omit,
        audio: Optional[Any] = omit,
        max_completion_tokens: Union[int, Omit] = omit,
        metadata: Union[Dict[str, str], Omit] = omit,
        modalities: Union[List[Any], Omit] = omit,
        prediction: Union[Any, Omit] = omit,
        reasoning_effort: Union[Any, Omit] = omit,
        store: Union[Optional[bool], Omit] = omit,
        **kwargs,
    ) -> Union[ChatCompletions, Iterator[ChatCompletionChunk]]: ...

    def stream_create(
        self,
        model,
        messages,
        stream,
        temperature,
        max_tokens,
        top_p,
        audio,
        max_completion_tokens,
        metadata,
        modalities,
        prediction,
        reasoning_effort,
        store,
        **kwargs,
    ) -> Union[ChatCompletions, Iterator[ChatCompletionChunk]]: ...

    def normal_create(
            self,
            model,
            messages,
            stream,
            temperature,
            max_tokens,
            top_p,
            audio,
            max_completion_tokens,
            metadata,
            modalities,
            prediction,
            reasoning_effort,
            store,
            **kwargs,
    ) -> ChatCompletions: ...


class ChatCompletion(APIResource, CustomStubBase):
    completions: Completions


class AsyncCompletions(AsyncAPIResource, CustomStubBase):
    def create(
        self,
        *,
        model: Optional[str] = "portkey-default",
        messages: Iterable[Any],
        stream: Union[bool, Omit] = omit,
        temperature: Union[float, Omit] = omit,
        max_tokens: Union[int, Omit] = omit,
        top_p: Union[float, Omit] = omit,
        audio: Optional[Any] = omit,
        max_completion_tokens: Union[int, Omit] = omit,
        metadata: Union[Dict[str, str], Omit] = omit,
        modalities: Union[List[Any], Omit] = omit,
        prediction: Union[Any, Omit] = omit,
        reasoning_effort: Union[Any, Omit] = omit,
        store: Union[Optional[bool], Omit] = omit,
        **kwargs,
    ) -> Union[ChatCompletions, AsyncIterator[ChatCompletionChunk]]: ...

    def stream_create(
        self,
        model,
        messages,
        stream,
        temperature,
        max_tokens,
        top_p,
        audio,
        max_completion_tokens,
        metadata,
        modalities,
        prediction,
        reasoning_effort,
        store,
        **kwargs,
    ) -> Union[ChatCompletions, AsyncIterator[ChatCompletionChunk]]: ...

    def normal_create(
        self,
        model,
        messages,
        stream,
        temperature,
        max_tokens,
        top_p,
        audio,
        max_completion_tokens,
        metadata,
        modalities,
        prediction,
        reasoning_effort,
        store,
        **kwargs,
    ) -> ChatCompletions: ...


class AsyncChatCompletion(AsyncAPIResource, CustomStubBase):
    completions: AsyncCompletions
