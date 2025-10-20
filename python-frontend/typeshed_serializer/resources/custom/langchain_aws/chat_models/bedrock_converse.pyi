from typing import Optional, Dict, Any, List, Callable, Literal, Mapping, Sequence

from langchain_core.caches import BaseCache
from langchain_core.callbacks import BaseCallbackManager, Callbacks
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter
from pydantic import SecretStr

from SonarPythonAnalyzerFakeStub import CustomStubBase

class ChatBedrockConverse(BaseChatModel):
    additional_model_request_fields: Dict[str, Any] | None = None
    additional_model_response_field_paths: List[str] | None = None
    aws_access_key_id: SecretStr | None
    aws_secret_access_key: SecretStr | None
    aws_session_token: SecretStr | None
    base_model_id: str | None = None
    base_model: str | None = None # Alias for base_model_id
    bedrock_client: Any = None
    cache: BaseCache | bool | None = None
    callback_manager: BaseCallbackManager | None = None
    callbacks: Callbacks = None
    client: Any = None
    config: Any = None
    credentials_profile_name: Optional[str] = None
    custom_get_token_ids: Callable[[str], list[int]] | None = None
    disable_streaming: bool | Literal['tool_calling'] = False
    endpoint_url: str | None = None
    base_url: str | None = None # Alias for endpoint_url
    guard_last_turn_only: bool = False
    guardrail_config: Dict[str, Any] | None = None
    guardrails: Dict[str, Any] | None = None # Alias for guardrail_config
    max_tokens: int | None = None
    metadata: dict[str, Any] | None = None
    model_id: str
    model: str # Alias for model_id
    performance_config: Mapping[str, Any] | None = None
    provider: str = ''
    rate_limiter: BaseRateLimiter | None = None
    raw_blocks: List[Dict[str, Any]] | None = None
    region_name: str | None = None
    request_metadata: Dict[str, str] | None = None
    stop_sequences: List[str] | None = None
    stop: List[str] | None = None # Alias for stop_sequences
    supports_tool_choice_values: Sequence[Literal['auto', 'any', 'tool']] | None = None
    tags: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    verbose: bool

