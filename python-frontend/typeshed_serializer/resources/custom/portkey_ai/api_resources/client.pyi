from portkey_ai.api_resources.base_client import APIClient, AsyncAPIClient

from SonarPythonAnalyzerFakeStub import CustomStubBase
from typing import Any, List, Mapping, Optional, Union
import httpx
from portkey_ai.api_resources.apis.chat_complete import ChatCompletion, AsyncChatCompletion


class Portkey(APIClient, CustomStubBase):
    completions: Any
    chat: ChatCompletion
    generations: Any
    prompts: Any
    embeddings: Any
    feedback: Any
    images: Any
    files: Any
    models: Any
    moderations: Any
    audio: Any
    batches: Any
    fine_tuning: Any
    vector_stores: Any
    responses: Any
    webhooks: Any
    evals: Any
    containers: Any
    admin: Any
    uploads: Any
    configs: Any
    api_keys: Any
    virtual_keys: Any
    logs: Any
    labels: Any
    collections: Any
    integrations: Any
    providers: Any
    guardrails: Any
    realtime: Any
    conversations: Any
    videos: Any
    beta: Any

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        virtual_key: Optional[str] = None,
        websocket_base_url: Optional[Union[str, httpx.URL]] = None,
        webhook_secret: Optional[str] = None,
        config: Optional[Union[Mapping, str]] = None,
        provider: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Union[Optional[dict[str, str]], str] = None,
        cache_namespace: Optional[str] = None,
        debug: Optional[bool] = None,
        cache_force_refresh: Optional[bool] = None,
        custom_host: Optional[str] = None,
        forward_headers: Optional[List[str]] = None,
        instrumentation: Optional[bool] = None,
        openai_project: Optional[str] = None,
        openai_organization: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: Optional[str] = None,
        vertex_project_id: Optional[str] = None,
        vertex_region: Optional[str] = None,
        workers_ai_account_id: Optional[str] = None,
        azure_resource_name: Optional[str] = None,
        azure_deployment_id: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_endpoint_name: Optional[str] = None,
        huggingface_base_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        request_timeout: Optional[int] = None,
        strict_open_ai_compliance: Optional[bool] = False,
        anthropic_beta: Optional[str] = None,
        anthropic_version: Optional[str] = None,
        mistral_fim_completion: Optional[str] = None,
        vertex_storage_bucket_name: Optional[str] = None,
        provider_file_name: Optional[str] = None,
        provider_model: Optional[str] = None,
        aws_s3_bucket: Optional[str] = None,
        aws_s3_object_key: Optional[str] = None,
        aws_bedrock_model: Optional[str] = None,
        fireworks_account_id: Optional[str] = None,
        calculate_audio_duration: Optional[bool] = True,
        **kwargs,
    ) -> None: ...


class AsyncPortkey(AsyncAPIClient, CustomStubBase):
    completions: Any
    chat: AsyncChatCompletion
    generations: Any
    prompts: Any
    embeddings: Any
    feedback: Any
    images: Any
    files: Any
    models: Any
    moderations: Any
    audio: Any
    batches: Any
    fine_tuning: Any
    vector_stores: Any
    responses: Any
    webhooks: Any
    evals: Any
    containers: Any
    admin: Any
    uploads: Any
    configs: Any
    api_keys: Any
    virtual_keys: Any
    logs: Any
    labels: Any
    collections: Any
    integrations: Any
    providers: Any
    guardrails: Any
    realtime: Any
    conversations: Any
    videos: Any
    beta: Any

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        virtual_key: Optional[str] = None,
        websocket_base_url: Optional[Union[str, httpx.URL]] = None,
        webhook_secret: Optional[str] = None,
        config: Optional[Union[Mapping, str]] = None,
        provider: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Union[Optional[dict[str, str]], str] = None,
        cache_namespace: Optional[str] = None,
        debug: Optional[bool] = None,
        cache_force_refresh: Optional[bool] = None,
        custom_host: Optional[str] = None,
        forward_headers: Optional[List[str]] = None,
        instrumentation: Optional[bool] = None,
        openai_project: Optional[str] = None,
        openai_organization: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: Optional[str] = None,
        vertex_project_id: Optional[str] = None,
        vertex_region: Optional[str] = None,
        workers_ai_account_id: Optional[str] = None,
        azure_resource_name: Optional[str] = None,
        azure_deployment_id: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_endpoint_name: Optional[str] = None,
        huggingface_base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        request_timeout: Optional[int] = None,
        strict_open_ai_compliance: Optional[bool] = False,
        anthropic_beta: Optional[str] = None,
        anthropic_version: Optional[str] = None,
        mistral_fim_completion: Optional[str] = None,
        vertex_storage_bucket_name: Optional[str] = None,
        provider_file_name: Optional[str] = None,
        provider_model: Optional[str] = None,
        aws_s3_bucket: Optional[str] = None,
        aws_s3_object_key: Optional[str] = None,
        aws_bedrock_model: Optional[str] = None,
        fireworks_account_id: Optional[str] = None,
        calculate_audio_duration: Optional[bool] = True,
        **kwargs,
    ) -> None: ...
