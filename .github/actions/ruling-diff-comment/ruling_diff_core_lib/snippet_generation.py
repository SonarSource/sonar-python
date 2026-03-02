from __future__ import annotations

from ruling_diff_core_lib.models_and_constants import (
    IssueDiff,
    MAX_SNIPPETS_PER_FILE,
    MAX_SNIPPETS_PER_RULE,
    SNIPPET_CONTEXT,
    OptionalSourceLines,
    RulingDiffIO,
    Snippet,
    SourceCache,
    SourceLines,
)


def unique_line_numbers_preserving_order(lines: list[int]) -> list[int]:
    return list(dict.fromkeys(lines))


def render_snippet(
    lines: OptionalSourceLines, issue_line: int, context: int = SNIPPET_CONTEXT
) -> str:
    if lines is None:
        return "(source file not found at this revision)"
    if issue_line == 0:
        return render_file_level_snippet(lines, context)
    return render_line_snippet(lines, issue_line, context)


def render_file_level_snippet(lines: SourceLines, context: int) -> str:
    if not lines:
        return ">>> FILE-LEVEL ISSUE\n(empty file)"
    end = min(len(lines), 1 + (2 * context))
    content = [f"    {index:>6} | {lines[index - 1]}" for index in range(1, end + 1)]
    return "\n".join([">>> FILE-LEVEL ISSUE", *content])


def render_line_snippet(lines: SourceLines, issue_line: int, context: int) -> str:
    if not lines:
        return f">>> ISSUE HERE (line {issue_line})\n(empty file)"
    clamped_line = max(1, min(issue_line, len(lines)))
    prefix = []
    if issue_line != clamped_line:
        prefix.append(
            f"(requested line {issue_line} not present, showing closest line {clamped_line})"
        )
    body = render_line_window(lines, clamped_line, context)
    return "\n".join(prefix + body)


def render_line_window(lines: SourceLines, center_line: int, context: int) -> list[str]:
    start = max(1, center_line - context)
    end = min(len(lines), center_line + context)
    rendered: list[str] = []
    for number in range(start, end + 1):
        marker = ">>>" if number == center_line else "   "
        rendered.append(f"{marker} {number:>6} | {lines[number - 1]}")
    return rendered


def build_snippets_for_rule(
    project: str,
    file_diffs: list[IssueDiff],
    source_cache: SourceCache,
    base_sha: str,
    head_sha: str,
    io: RulingDiffIO,
) -> list[Snippet]:
    snippets: list[Snippet] = []
    for file_diff in file_diffs:
        snippets.extend(
            collect_snippets_for_file(
                project=project,
                file_diff=file_diff,
                source_cache=source_cache,
                base_sha=base_sha,
                head_sha=head_sha,
                io=io,
            )
        )
        if len(snippets) >= MAX_SNIPPETS_PER_RULE:
            break
    return snippets[:MAX_SNIPPETS_PER_RULE]


def collect_snippets_for_file(
    *,
    project: str,
    file_diff: IssueDiff,
    source_cache: SourceCache,
    base_sha: str,
    head_sha: str,
    io: RulingDiffIO,
) -> list[Snippet]:
    snippets: list[Snippet] = []
    for change_kind, lines in (
        ("removed", unique_line_numbers_preserving_order(file_diff.removed_lines)),
        ("added", unique_line_numbers_preserving_order(file_diff.added_lines)),
    ):
        for line_number in lines[:MAX_SNIPPETS_PER_FILE]:
            snippets.append(
                create_issue_snippet(
                    project=project,
                    file_path=file_diff.file_path,
                    line_number=line_number,
                    change_kind=change_kind,
                    source_cache=source_cache,
                    base_sha=base_sha,
                    head_sha=head_sha,
                    io=io,
                )
            )
    return snippets


def create_issue_snippet(
    *,
    project: str,
    file_path: str,
    line_number: int,
    change_kind: str,
    source_cache: SourceCache,
    base_sha: str,
    head_sha: str,
    io: RulingDiffIO,
) -> Snippet:
    ref = head_sha if change_kind == "added" else base_sha
    source_path = io.resolve_source_path(project, file_path)
    lines = load_source_lines_with_cache(source_cache, ref, source_path, io)
    body = (
        f"(source file not found at this revision: {file_path})"
        if lines is None
        else render_snippet(lines, line_number)
    )
    return Snippet(
        file_path=file_path,
        line_number=line_number,
        change_kind=change_kind,
        body=body,
    )


def load_source_lines_with_cache(
    cache: SourceCache,
    ref: str,
    path: str,
    io: RulingDiffIO,
) -> OptionalSourceLines:
    key = (ref, path)
    if key not in cache:
        content = io.load_text_at_ref(path, ref)
        cache[key] = None if content is None else content.splitlines()
    return cache[key]
