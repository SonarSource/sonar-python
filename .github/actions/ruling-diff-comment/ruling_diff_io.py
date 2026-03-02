from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from ruling_diff_core_lib.models_and_constants import (
    COMMENT_MARKER,
    EXPECTED_RULING_ROOT,
    PROJECT_SOURCE_OVERRIDES,
)

RULING_SOURCES_SUBMODULE = "private/its-enterprise/sources_ruling"
SOURCES_INTERNAL_RULING_ROOT = "private/its-enterprise/sources_internal_ruling"
SOURCES_INTERNAL_NAMESPACE_RULING_ROOT = (
    "private/its-enterprise/sources_internal_namespace_ruling"
)


class CommandError(RuntimeError):
    pass


class GitHubActionIO:
    def load_json_at_ref(self, path: str, ref: str) -> dict[str, list[int]] | None:
        return load_json_at_ref(path, ref)

    def load_text_at_ref(self, path: str, ref: str) -> str | None:
        return load_text_at_ref(path, ref)

    def resolve_source_path(self, project: str, file_path: str) -> str:
        if project == "project":
            return self._resolve_project_source_path(file_path)
        source_root = PROJECT_SOURCE_OVERRIDES.get(
            project, f"{RULING_SOURCES_SUBMODULE}/{project}"
        )
        return f"{source_root}/{file_path.lstrip('/')}"

    def _resolve_project_source_path(self, file_path: str) -> str:
        clean_path = file_path.lstrip("/")
        primary_candidate = f"{RULING_SOURCES_SUBMODULE}/{clean_path}"
        candidates = [primary_candidate]
        candidates.extend(
            self._with_direct_children_prefixes(RULING_SOURCES_SUBMODULE, clean_path)
        )
        candidates.append(f"{SOURCES_INTERNAL_RULING_ROOT}/{clean_path}")
        candidates.append(f"{SOURCES_INTERNAL_NAMESPACE_RULING_ROOT}/{clean_path}")
        candidates.extend(
            self._with_direct_children_prefixes(
                SOURCES_INTERNAL_NAMESPACE_RULING_ROOT, clean_path
            )
        )
        for candidate in candidates:
            if Path(candidate).is_file():
                return candidate
        return primary_candidate

    def _with_direct_children_prefixes(self, root: str, file_path: str) -> list[str]:
        root_path = Path(root)
        if not root_path.is_dir():
            return []
        return [
            f"{root}/{child.name}/{file_path}"
            for child in sorted(root_path.iterdir(), key=lambda path: path.name)
            if child.is_dir() and not child.name.startswith(".")
        ]


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise CommandError(
            format_command_failure(
                command, result.stdout, result.stderr, result.returncode
            )
        )
    return result.stdout


def format_command_failure(
    command: list[str], stdout: str, stderr: str, returncode: int
) -> str:
    return (
        f"Command failed with exit code {returncode}: {' '.join(command)}\n"
        f"stdout: {stdout}\n"
        f"stderr: {stderr}"
    )


def run_gh_json(command: list[str]) -> dict | list:
    output = run_command(["gh", *command])
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise CommandError(f"Could not parse JSON from gh output: {exc}") from exc


def run_gh_paginated_items(endpoint: str) -> list[dict]:
    output = run_command(["gh", "api", "--paginate", endpoint])
    docs = parse_json_documents(output)
    items: list[dict] = []
    for doc in docs:
        if not isinstance(doc, list):
            raise CommandError("Unexpected response type while listing paginated items")
        for item in doc:
            if isinstance(item, dict):
                items.append(item)
    return items


def parse_json_documents(content: str) -> list[object]:
    decoder = json.JSONDecoder()
    index = 0
    documents: list[object] = []
    while index < len(content):
        while index < len(content) and content[index].isspace():
            index += 1
        if index >= len(content):
            break
        document, next_index = decoder.raw_decode(content, index)
        documents.append(document)
        index = next_index
    return documents


def get_changed_ruling_files(base_sha: str, head_sha: str) -> list[str]:
    output = run_command(
        [
            "git",
            "diff",
            "--name-only",
            f"{base_sha}...{head_sha}",
            "--",
            f"{EXPECTED_RULING_ROOT}/",
        ]
    )
    changed = [
        path
        for path in (line.strip() for line in output.splitlines())
        if is_ruling_json(path)
    ]
    return sorted(set(changed))


def is_ruling_json(path: str) -> bool:
    return (
        bool(path)
        and path.endswith(".json")
        and path.startswith(f"{EXPECTED_RULING_ROOT}/")
    )


def _is_missing_at_ref(stderr: str) -> bool:
    return any(
        marker in stderr
        for marker in ("exists on disk, but not in", "does not exist in", "path '")
    )


def load_json_at_ref(path: str, ref: str) -> dict[str, list[int]] | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, text=True
    )
    if result.returncode != 0:
        if _is_missing_at_ref(result.stderr):
            return None
        raise CommandError(
            f"Failed to read file at ref: git show {ref}:{path}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return parse_ruling_json(result.stdout, path, ref)


def parse_ruling_json(content: str, path: str, ref: str) -> dict[str, list[int]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path} at {ref}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Ruling file {path} at {ref} must be a JSON object")
    return normalize_ruling_json(data, path, ref)


def normalize_ruling_json(data: dict, path: str, ref: str) -> dict[str, list[int]]:
    normalized: dict[str, list[int]] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"Ruling file {path} at {ref} has non-string key")
        if not isinstance(value, list) or not all(isinstance(v, int) for v in value):
            raise ValueError(
                f"Ruling file {path} at {ref} has non-integer line list for key {key}"
            )
        normalized[key] = value
    return normalized


def load_text_at_ref(path: str, ref: str) -> str | None:
    if is_ruling_source_path(path):
        return load_submodule_text_at_ref(path, ref)

    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout
    if _is_missing_at_ref(result.stderr):
        return load_text_with_workspace_fallback(path, ref)
    raise CommandError(
        f"Failed to read source file at ref: git show {ref}:{path}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def is_ruling_source_path(path: str) -> bool:
    return path.startswith(f"{RULING_SOURCES_SUBMODULE}/")


def load_submodule_text_at_ref(path: str, ref: str) -> str | None:
    submodule_commit = get_submodule_commit_for_ref(ref)
    if submodule_commit is None:
        return load_text_with_workspace_fallback(path, ref)

    submodule_relative_path = path[len(f"{RULING_SOURCES_SUBMODULE}/") :]
    content = read_submodule_file_at_commit(submodule_commit, submodule_relative_path)
    if content is not None:
        return content

    fetch_submodule_commit(submodule_commit)
    content = read_submodule_file_at_commit(submodule_commit, submodule_relative_path)
    if content is not None:
        return content

    logging.warning(
        "Source file '%s' not found in submodule commit %s for %s",
        path,
        submodule_commit,
        ref,
    )
    return load_text_with_workspace_fallback(path, ref)


def get_submodule_commit_for_ref(ref: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", f"{ref}:{RULING_SOURCES_SUBMODULE}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logging.warning(
            "Could not resolve ruling sources submodule commit for %s: %s",
            ref,
            result.stderr.strip(),
        )
        return None
    return result.stdout.strip()


def read_submodule_file_at_commit(commit: str, relative_path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", RULING_SOURCES_SUBMODULE, "show", f"{commit}:{relative_path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout
    return None


def fetch_submodule_commit(commit: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            RULING_SOURCES_SUBMODULE,
            "fetch",
            "--depth",
            "1",
            "origin",
            commit,
        ],
        capture_output=True,
        text=True,
    )


def load_text_with_workspace_fallback(path: str, ref: str) -> str | None:
    workspace_content = load_workspace_text(path)
    if workspace_content is None:
        logging.warning("Source file '%s' not found at %s", path, ref)
        return None
    logging.warning("Source file '%s' not found at %s, using workspace copy", path, ref)
    return workspace_content


def load_workspace_text(path: str) -> str | None:
    workspace_path = Path(path)
    if not workspace_path.is_file():
        return None
    return workspace_path.read_text(encoding="utf-8")


def get_existing_comment_id(pr_number: str, repository: str) -> str | None:
    comments = run_gh_paginated_items(
        f"repos/{repository}/issues/{pr_number}/comments?per_page=100"
    )
    for comment in comments:
        if COMMENT_MARKER in comment.get("body", ""):
            return str(comment["id"])
    return None


def post_or_update_comment(pr_number: str, repository: str, body: str) -> None:
    comment_id = get_existing_comment_id(pr_number, repository)
    if comment_id is None:
        logging.info("Posting new ruling diff comment on PR #%s", pr_number)
        run_command(
            ["gh", "pr", "comment", pr_number, "--repo", repository, "--body", body]
        )
        return
    logging.info(
        "Updating existing ruling diff comment %s on PR #%s", comment_id, pr_number
    )
    run_command(
        [
            "gh",
            "api",
            "--method",
            "PATCH",
            f"repos/{repository}/issues/comments/{comment_id}",
            "-f",
            f"body={body}",
        ]
    )
