from pathlib import Path
from typing import Literal, Any, Callable
from claude_agent_sdk.types import (
    ToolsPreset,
    McpServerConfig,
    PermissionMode,
    SdkBeta,
    SettingSource,
    HookEvent,
    HookMatcher,
    CanUseTool,
    AgentDefinition,
    SandboxSettings,
    SdkPluginConfig,
    ThinkingConfig,
)

from SonarPythonAnalyzerFakeStub import CustomStubBase

class SystemPromptPreset(CustomStubBase):
    def __init__(
            self,
            type: Literal["preset"],
            preset: Literal["claude_code"],
            append: str,
    ) -> None: ...

class ClaudeAgentOptions(CustomStubBase):
    def __init__(
        self,
        tools: list[str] | ToolsPreset | None,
        allowed_tools: list[str],
        system_prompt: str | SystemPromptPreset | None,
        mcp_servers: dict[str, McpServerConfig] | str | Path,
        permission_mode: PermissionMode | None,
        continue_conversation: bool,
        resume: str | None,
        max_turns: int | None,
        max_budget_usd: float | None,
        disallowed_tools: list[str],
        model: str | None,
        fallback_model: str | None,
        betas: list[SdkBeta],
        permission_prompt_tool_name: str | None,
        cwd: str | Path | None,
        cli_path: str | Path | None,
        settings: str | None,
        add_dirs: list[str | Path],
        env: dict[str, str],
        extra_args: dict[str, str | None],
        max_buffer_size: int | None,
        debug_stderr: Any,
        stderr: Callable[[str], None] | None,
        can_use_tool: CanUseTool | None,
        hooks: dict[HookEvent, list[HookMatcher]] | None,
        user: str | None,
        include_partial_messages: bool,
        fork_session: bool,
        agents: dict[str, AgentDefinition] | None,
        setting_sources: list[SettingSource] | None,
        sandbox: SandboxSettings | None,
        plugins: list[SdkPluginConfig],
        max_thinking_tokens: int | None,
        thinking: ThinkingConfig | None,
        effort: Literal["low", "medium", "high", "max"] | None,
        output_format: dict[str, Any] | None,
        enable_file_checkpointing: bool,
    ) -> None: ...

