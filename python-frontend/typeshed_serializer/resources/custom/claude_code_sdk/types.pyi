from SonarPythonAnalyzerFakeStub import CustomStubBase
from pathlib import Path
from typing import Any, Literal

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]


class ClaudeCodeOptions(CustomStubBase):
    def __init__(
            self,
            allowed_tools: list[str],
            system_prompt: str | None,
            append_system_prompt: str | None,
            mcp_servers: dict[str, McpServerConfig] | str | Path,
            permission_mode: PermissionMode | None,
            continue_conversation: bool,
            resume: str | None,
            max_turns: int | None,
            disallowed_tools: list[str],
            model: str | None,
            permission_prompt_tool_name: str | None,
            cwd: str | Path | None,
            settings: str | None,
            add_dirs: list[str | Path],
            env: dict[str, str],
            extra_args: dict[str, str | None],
            debug_stderr: Any,
            can_use_tool: CanUseTool | None,
            hooks: dict[HookEvent, list[HookMatcher]] | None,
            user: str | None,
            include_partial_messages: bool) -> ClaudeCodeOptions: ...
