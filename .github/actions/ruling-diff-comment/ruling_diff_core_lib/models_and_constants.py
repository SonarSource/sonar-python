from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

RulingJson = dict[str, list[int]]
OptionalRulingJson = RulingJson | None
SourceLines = list[str]
OptionalSourceLines = SourceLines | None
SourceCache = dict[tuple[str, str], OptionalSourceLines]


class RulingDiffIO(Protocol):
    def load_json_at_ref(self, path: str, ref: str) -> OptionalRulingJson:
        ...

    def load_text_at_ref(self, path: str, ref: str) -> str | None:
        ...

    def resolve_source_path(self, project: str, file_path: str) -> str:
        ...

EXPECTED_RULING_ROOT = (
    "private/its-enterprise/ruling/src/test/resources/expected_ruling"
)
COMMENT_MARKER = "<!-- ruling-diff-comment -->"
COMMENT_SOFT_LIMIT = 60000
SNIPPET_CONTEXT = 5
MAX_SNIPPETS_PER_FILE = 3
MAX_SNIPPETS_PER_RULE = 15

PROJECT_SOURCE_OVERRIDES = {
    "buildbot": "private/its-enterprise/sources_ruling/buildbot-0.8.6p1",
    "buildbot-slave": "private/its-enterprise/sources_ruling/buildbot-slave-0.8.6p1",
    "django": "private/its-enterprise/sources_ruling/django-2.2.3",
    "django-cms": "private/its-enterprise/sources_ruling/django-cms-3.7.1",
    "docker-compose": "private/its-enterprise/sources_ruling/docker-compose-1.24.1",
    "mypy": "private/its-enterprise/sources_ruling/mypy-0.782",
    "numpy": "private/its-enterprise/sources_ruling/numpy-1.16.4",
    "tornado": "private/its-enterprise/sources_ruling/tornado-2.3",
    "twisted": "private/its-enterprise/sources_ruling/twisted-12.1.0",
    "sources_internal_ruling": "private/its-enterprise/sources_internal_ruling",
    "namespace_basic": "private/its-enterprise/sources_internal_namespace_ruling/basic_namespace",
    "namespace_mixed": "private/its-enterprise/sources_internal_namespace_ruling/mixed_namespace",
}


@dataclass(frozen=True)
class IssueDiff:
    file_path: str
    added_lines: list[int]
    removed_lines: list[int]


@dataclass(frozen=True)
class Snippet:
    file_path: str
    line_number: int
    change_kind: str
    body: str


@dataclass(frozen=True)
class RuleDiff:
    project: str
    repo: str
    rule_key: str
    file_diffs: list[IssueDiff]
    snippets: list[Snippet]
    is_new_file: bool
    is_deleted_file: bool
