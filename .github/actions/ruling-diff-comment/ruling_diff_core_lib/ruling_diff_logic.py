from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath

from ruling_diff_core_lib.models_and_constants import (
    EXPECTED_RULING_ROOT,
    IssueDiff,
    OptionalRulingJson,
    RulingDiffIO,
    RuleDiff,
    RulingJson,
    SourceCache,
    Snippet,
)
from ruling_diff_core_lib.snippet_generation import build_snippets_for_rule


def parse_ruling_path(path: str) -> tuple[str, str, str]:
    prefix = f"{EXPECTED_RULING_ROOT}/"
    if not path.startswith(prefix):
        raise ValueError(f"Unexpected ruling path outside expected root: {path}")
    relative_path = path[len(prefix) :]
    project, filename = parse_ruling_relative_path(relative_path)
    repository, rule_key = parse_rule_filename(filename)
    return project, repository, rule_key


def parse_ruling_relative_path(relative_path: str) -> tuple[str, str]:
    parts = PurePosixPath(relative_path).parts
    if len(parts) != 2:
        raise ValueError(
            f"Expected '<project>/<repo>-<rule>.json' path, got: {relative_path}"
        )
    return parts[0], parts[1]


def parse_rule_filename(filename: str) -> tuple[str, str]:
    if not filename.endswith(".json"):
        raise ValueError(f"Expected json filename, got: {filename}")
    basename = filename[:-5]
    if "-" not in basename:
        raise ValueError(f"Expected '<repo>-<ruleKey>.json', got: {filename}")
    repository, rule_key = basename.rsplit("-", 1)
    if not repository:
        raise ValueError(f"Missing repo in filename: {filename}")
    if not rule_key.startswith("S") or not rule_key[1:].isdigit():
        raise ValueError(f"Invalid rule key in filename: {filename}")
    return repository, rule_key


def strip_project_key(key: str) -> str:
    return key.split(":", 1)[1] if ":" in key else key


def diff_ruling_jsons(
    old: OptionalRulingJson, new: OptionalRulingJson
) -> list[IssueDiff]:
    old_map = old or {}
    new_map = new or {}
    return [
        issue_diff
        for issue_diff in (
            diff_single_file_key(key, old_map, new_map)
            for key in sorted(set(old_map) | set(new_map))
        )
        if issue_diff is not None
    ]


def diff_single_file_key(
    key: str, old_map: RulingJson, new_map: RulingJson
) -> IssueDiff | None:
    old_counter = Counter(old_map.get(key, []))
    new_counter = Counter(new_map.get(key, []))
    added_lines: list[int] = expand_line_counter(new_counter - old_counter)
    removed_lines: list[int] = expand_line_counter(old_counter - new_counter)
    if not added_lines and not removed_lines:
        return None
    return IssueDiff(
        file_path=strip_project_key(key),
        added_lines=added_lines,
        removed_lines=removed_lines,
    )


def expand_line_counter(counter: Counter[int]) -> list[int]:
    line_numbers: list[int] = []
    for line_number in sorted(counter):
        line_numbers.extend([line_number] * counter[line_number])
    return line_numbers


def build_rule_diffs(
    changed_files: list[str],
    base_sha: str,
    head_sha: str,
    io: RulingDiffIO,
) -> list[RuleDiff]:
    source_cache: SourceCache = {}
    diffs = [
        build_rule_diff_for_file(
            path,
            base_sha,
            head_sha,
            source_cache,
            io,
        )
        for path in sorted(changed_files)
    ]
    return sorted(
        [rule_diff for rule_diff in diffs if rule_diff is not None],
        key=lambda diff: (diff.project, diff.repo, diff.rule_key),
    )


def build_rule_diff_for_file(
    path: str,
    base_sha: str,
    head_sha: str,
    source_cache: SourceCache,
    io: RulingDiffIO,
) -> RuleDiff | None:
    project, repository, rule_key = parse_ruling_path(path)
    old_json: OptionalRulingJson = io.load_json_at_ref(path, base_sha)
    new_json: OptionalRulingJson = io.load_json_at_ref(path, head_sha)
    file_diffs: list[IssueDiff] = diff_ruling_jsons(old_json, new_json)
    if not file_diffs:
        return None
    snippets: list[Snippet] = build_snippets_for_rule(
        project,
        file_diffs,
        source_cache,
        base_sha,
        head_sha,
        io,
    )
    return RuleDiff(
        project=project,
        repo=repository,
        rule_key=rule_key,
        file_diffs=file_diffs,
        snippets=snippets,
        is_new_file=old_json is None,
        is_deleted_file=new_json is None,
    )
