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
    "private/its/ruling/src/test/resources/expected"
)
COMMENT_MARKER = "<!-- ruling-diff-comment -->"
COMMENT_SOFT_LIMIT = 60000
SNIPPET_CONTEXT = 5
MAX_SNIPPETS_PER_FILE = 3
MAX_SNIPPETS_PER_RULE = 15

PROJECT_SOURCE_OVERRIDES = {
    "sources_internal_ruling": "private/its/sources_internal_ruling",
    "namespace_basic": "private/its/sources_internal_namespace_ruling/basic_namespace",
    "namespace_mixed": "private/its/sources_internal_namespace_ruling/mixed_namespace",
    "addInitPy_base": "private/its/sources_pr_analysis/addInitPy_base",
    "addInitPy_pr": "private/its/sources_pr_analysis/addInitPy_pr",
    "baseCommit": "private/its/sources_pr_analysis/baseCommit",
    "changeInImportedModule": "private/its/sources_pr_analysis/changeInImportedModule",
    "changeInPackageInit": "private/its/sources_pr_analysis/changeInPackageInit",
    "changeInParent": "private/its/sources_pr_analysis/changeInParent",
    "changeInRelativeImport": "private/its/sources_pr_analysis/changeInRelativeImport",
    "deletedFile": "private/its/sources_pr_analysis/deletedFile",
    "djangoViewsScenario": "private/its/sources_pr_analysis/djangoViewsScenario",
    "duplication": "private/its/sources_pr_analysis/duplication",
    "newFile": "private/its/sources_pr_analysis/newFile",
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
