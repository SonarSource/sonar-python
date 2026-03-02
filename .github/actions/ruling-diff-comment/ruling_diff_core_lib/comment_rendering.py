from __future__ import annotations

from ruling_diff_core_lib.models_and_constants import (
    COMMENT_MARKER,
    COMMENT_SOFT_LIMIT,
    RuleDiff,
    Snippet,
)


def format_comment(
    rule_diffs: list[RuleDiff], soft_limit: int = COMMENT_SOFT_LIMIT
) -> str:
    if not rule_diffs:
        return "\n".join(
            [
                COMMENT_MARKER,
                "## Ruling Diff Summary",
                "",
                "No issue deltas detected."
            ]
        )
    comment = format_comment_header(rule_diffs)
    return append_rule_sections_with_soft_limit(comment, rule_diffs, soft_limit)


def format_comment_header(rule_diffs: list[RuleDiff]) -> str:
    added = sum(len(diff.added_lines) for rule in rule_diffs for diff in rule.file_diffs)
    removed = sum(
        len(diff.removed_lines) for rule in rule_diffs for diff in rule.file_diffs
    )
    return "\n".join(
        [
            COMMENT_MARKER,
            "## Ruling Diff Summary",
            "",
            f"Detected changes in {len(rule_diffs)} rule files: {removed} issues removed, {added} issues added.",
            "",
        ]
    )


def append_rule_sections_with_soft_limit(
    comment: str, rule_diffs: list[RuleDiff], soft_limit: int
) -> str:
    sections = [format_rule_section(rule_diff) for rule_diff in rule_diffs]
    accepted_sections: list[str] = []
    truncated_count = 0
    for index, section in enumerate(sections):
        candidate = comment + "\n\n".join(accepted_sections + [section])
        if len(candidate) > soft_limit:
            truncated_count = len(sections) - index
            break
        accepted_sections.append(section)
    if accepted_sections:
        comment += "\n\n".join(accepted_sections)
    if truncated_count:
        comment = append_truncation_notice(comment, truncated_count, bool(accepted_sections))
    return comment


def append_truncation_notice(comment: str, count: int, has_sections: bool) -> str:
    separator = "\n\n" if has_sections else ""
    return (
        comment
        + separator
        + f"... and {count} more rules with changes (diff too large to display fully)"
    )


def format_rule_section(rule_diff: RuleDiff) -> str:
    lines = ["<details>", f"<summary>{format_rule_summary(rule_diff)}</summary>", ""]
    if not rule_diff.snippets:
        lines.append("No source snippets available for this rule.")
    else:
        for snippet in rule_diff.snippets:
            lines.append(format_snippet_block(snippet))
            lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def format_rule_summary(rule_diff: RuleDiff) -> str:
    removed = sum(len(diff.removed_lines) for diff in rule_diff.file_diffs)
    added = sum(len(diff.added_lines) for diff in rule_diff.file_diffs)
    summary_parts = [
        f"<b>{rule_diff.rule_key}</b> (<code>{rule_diff.repo}</code>) on <b>{rule_diff.project}</b>",
        f"{removed} issues removed, {added} issues added",
    ]
    if rule_diff.is_new_file:
        summary_parts.append("new ruling file")
    if rule_diff.is_deleted_file:
        summary_parts.append("deleted ruling file")
    return " - ".join(summary_parts)


def format_snippet_block(snippet: Snippet) -> str:
    return "\n".join([format_snippet_header(snippet), "```python", snippet.body, "```"])


def format_snippet_header(snippet: Snippet) -> str:
    label = "Added" if snippet.change_kind == "added" else "Removed"
    location = "file-level" if snippet.line_number == 0 else f"line {snippet.line_number}"
    return f"**{label}** `{snippet.file_path}` ({location})"
