import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE_DIR = pathlib.Path(__file__).parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import ruling_diff_core as core
import ruling_diff_io as io


IssueDiff = core.IssueDiff
RuleDiff = core.RuleDiff
Snippet = core.Snippet


class FakeRulingDiffIO:
    def __init__(
        self,
        json_by_ref_path: dict[tuple[str, str], dict[str, list[int]] | None],
        text_by_ref_path: dict[tuple[str, str], str | None],
    ) -> None:
        self.json_by_ref_path = json_by_ref_path
        self.text_by_ref_path = text_by_ref_path
        self.load_json_calls: list[tuple[str, str]] = []
        self.load_text_calls: list[tuple[str, str]] = []
        self.resolve_calls: list[tuple[str, str]] = []

    def load_json_at_ref(self, path: str, ref: str) -> dict[str, list[int]] | None:
        self.load_json_calls.append((path, ref))
        return self.json_by_ref_path.get((path, ref))

    def load_text_at_ref(self, path: str, ref: str) -> str | None:
        self.load_text_calls.append((path, ref))
        return self.text_by_ref_path.get((path, ref))

    def resolve_source_path(self, project: str, file_path: str) -> str:
        self.resolve_calls.append((project, file_path))
        return f"sources/{project}/{file_path.lstrip('/')}"


class ParsePathTest(unittest.TestCase):
    def test_parse_ruling_path(self) -> None:
        path = "private/its-enterprise/ruling/src/test/resources/expected_ruling/airflow/python-S1066.json"
        self.assertEqual(("airflow", "python", "S1066"), core.parse_ruling_path(path))

    def test_parse_ruling_path_with_pythonenterprise(self) -> None:
        path = "private/its-enterprise/ruling/src/test/resources/expected_ruling/specific-rules/pythonenterprise-S7471.json"
        self.assertEqual(
            ("specific-rules", "pythonenterprise", "S7471"),
            core.parse_ruling_path(path),
        )


class DiffLogicTest(unittest.TestCase):
    def test_diff_ruling_jsons_added_issues(self) -> None:
        diffs = core.diff_ruling_jsons({"proj:a.py": [1, 2]}, {"proj:a.py": [1, 2, 3]})
        self.assertEqual(1, len(diffs))
        self.assertEqual("a.py", diffs[0].file_path)
        self.assertEqual([3], diffs[0].added_lines)

    def test_diff_ruling_jsons_removed_issues(self) -> None:
        diffs = core.diff_ruling_jsons({"proj:a.py": [1, 2, 3]}, {"proj:a.py": [1]})
        self.assertEqual([2, 3], diffs[0].removed_lines)

    def test_diff_ruling_jsons_new_file_entry(self) -> None:
        diffs = core.diff_ruling_jsons(
            {"proj:a.py": [1]},
            {"proj:a.py": [1], "proj:b.py": [5]},
        )
        self.assertEqual("b.py", diffs[0].file_path)
        self.assertEqual([5], diffs[0].added_lines)

    def test_diff_ruling_jsons_removed_file_entry(self) -> None:
        diffs = core.diff_ruling_jsons(
            {"proj:a.py": [1], "proj:b.py": [5]},
            {"proj:a.py": [1]},
        )
        self.assertEqual("b.py", diffs[0].file_path)
        self.assertEqual([5], diffs[0].removed_lines)

    def test_diff_ruling_jsons_new_ruling_file(self) -> None:
        diffs = core.diff_ruling_jsons(None, {"proj:a.py": [10]})
        self.assertEqual([10], diffs[0].added_lines)

    def test_diff_ruling_jsons_deleted_ruling_file(self) -> None:
        diffs = core.diff_ruling_jsons({"proj:a.py": [10]}, None)
        self.assertEqual([10], diffs[0].removed_lines)

    def test_diff_ruling_jsons_no_changes(self) -> None:
        self.assertEqual(
            [],
            core.diff_ruling_jsons(
                {"proj:a.py": [10], "proj:b.py": [11, 12]},
                {"proj:a.py": [10], "proj:b.py": [11, 12]},
            ),
        )

    def test_duplicate_line_numbers_preserved(self) -> None:
        diffs = core.diff_ruling_jsons({"proj:a.py": [297]}, {"proj:a.py": [297, 297]})
        self.assertEqual([297], diffs[0].added_lines)


class FormattingTest(unittest.TestCase):
    def test_format_comment_single_rule(self) -> None:
        rule_diff = RuleDiff(
            project="airflow",
            repo="python",
            rule_key="S107",
            file_diffs=[IssueDiff("airflow/hooks/a.py", [10, 11], [8])],
            snippets=[
                Snippet(
                    file_path="airflow/hooks/a.py",
                    line_number=10,
                    change_kind="added",
                    body=">>>     10 | x = 1",
                )
            ],
            is_new_file=False,
            is_deleted_file=False,
        )
        comment = core.format_comment([rule_diff])
        self.assertIn("## Ruling Diff Summary", comment)
        self.assertIn("<details>", comment)
        self.assertIn("**Added** `airflow/hooks/a.py` (line 10)", comment)
        self.assertIn(">>>     10 | x = 1", comment)
        self.assertIn("```python", comment)

    def test_format_comment_multiple_rules(self) -> None:
        rule_diffs = [
            RuleDiff(
                project="airflow",
                repo="python",
                rule_key="S107",
                file_diffs=[IssueDiff("airflow/hooks/a.py", [10], [])],
                snippets=[
                    Snippet("airflow/hooks/a.py", 10, "added", ">>>     10 | return 1")
                ],
                is_new_file=False,
                is_deleted_file=False,
            ),
            RuleDiff(
                project="django",
                repo="python",
                rule_key="S3699",
                file_diffs=[IssueDiff("django/core/b.py", [20], [15])],
                snippets=[
                    Snippet(
                        "django/core/b.py", 15, "removed", ">>>     15 | return None"
                    )
                ],
                is_new_file=False,
                is_deleted_file=False,
            ),
        ]
        comment = core.format_comment(rule_diffs)
        self.assertIn("Detected changes in 2 rule files", comment)
        self.assertIn("S107", comment)
        self.assertIn("S3699", comment)

    def test_format_comment_respects_collapse(self) -> None:
        rule_diff = RuleDiff(
            project="airflow",
            repo="python",
            rule_key="S107",
            file_diffs=[IssueDiff("airflow/hooks/a.py", [10], [])],
            snippets=[
                Snippet("airflow/hooks/a.py", 10, "added", ">>>     10 | return 1")
            ],
            is_new_file=False,
            is_deleted_file=False,
        )
        comment = core.format_comment([rule_diff])
        self.assertIn("<details>", comment)
        self.assertIn("</details>", comment)

    def test_strip_project_key_from_path(self) -> None:
        self.assertEqual(
            "airflow/foo.py", core.strip_project_key("airflow:airflow/foo.py")
        )

    def test_line_zero_displayed_as_file_level(self) -> None:
        rule_diff = RuleDiff(
            project="specific-rules",
            repo="python",
            rule_key="S1451",
            file_diffs=[IssueDiff("S1716.py", [0], [0])],
            snippets=[Snippet("S1716.py", 0, "added", ">>> FILE-LEVEL ISSUE")],
            is_new_file=False,
            is_deleted_file=False,
        )
        comment = core.format_comment([rule_diff])
        self.assertIn("file-level", comment)
        self.assertIn(">>> FILE-LEVEL ISSUE", comment)

    def test_format_comment_truncates_when_limit_reached(self) -> None:
        rule_diffs = [
            RuleDiff(
                project=f"project-{index}",
                repo="python",
                rule_key=f"S{1000 + index}",
                file_diffs=[IssueDiff("a.py", [1], [2])],
                snippets=[
                    Snippet(
                        "a.py", 1, "added", "\n".join([f"line {i}" for i in range(50)])
                    )
                ],
                is_new_file=False,
                is_deleted_file=False,
            )
            for index in range(5)
        ]
        comment = core.format_comment(rule_diffs, soft_limit=500)
        self.assertIn("diff too large to display fully", comment)


class SnippetRenderingTest(unittest.TestCase):
    def test_render_line_snippet_uses_plus_minus_five_lines(self) -> None:
        lines = [f"line {i}" for i in range(1, 21)]
        rendered = core.render_line_snippet(lines, issue_line=10, context=5)
        self.assertIn("      5 | line 5", rendered)
        self.assertIn(">>>     10 | line 10", rendered)
        self.assertIn("     15 | line 15", rendered)

    def test_render_line_snippet_handles_out_of_range_line(self) -> None:
        rendered = core.render_line_snippet(["alpha", "beta"], issue_line=99, context=5)
        self.assertIn("requested line 99 not present", rendered)
        self.assertIn(">>>      2 | beta", rendered)

    def test_render_file_level_snippet_marker(self) -> None:
        rendered = core.render_file_level_snippet(["a", "b", "c"], context=5)
        self.assertIn(">>> FILE-LEVEL ISSUE", rendered)

    def test_render_snippet_missing_source_placeholder(self) -> None:
        rendered = core.render_snippet(None, issue_line=12, context=5)
        self.assertEqual("(source file not found at this revision)", rendered)


class BuildRuleDiffsWithIOTest(unittest.TestCase):
    def test_build_rule_diffs_uses_io_object_and_respects_refs(self) -> None:
        changed_file = (
            "private/its-enterprise/ruling/src/test/resources/expected_ruling/"
            "airflow/python-S107.json"
        )
        io_impl = FakeRulingDiffIO(
            json_by_ref_path={
                (changed_file, "base-sha"): {"airflow:a.py": [2]},
                (changed_file, "head-sha"): {"airflow:a.py": [2, 7]},
            },
            text_by_ref_path={
                ("sources/airflow/a.py", "head-sha"): "\n".join(
                    [f"line {index}" for index in range(1, 12)]
                ),
            },
        )

        diffs = core.build_rule_diffs([changed_file], "base-sha", "head-sha", io_impl)

        self.assertEqual(1, len(diffs))
        self.assertEqual("airflow", diffs[0].project)
        self.assertEqual("python", diffs[0].repo)
        self.assertEqual("S107", diffs[0].rule_key)
        self.assertEqual([7], diffs[0].file_diffs[0].added_lines)
        self.assertEqual([], diffs[0].file_diffs[0].removed_lines)
        self.assertIn((changed_file, "base-sha"), io_impl.load_json_calls)
        self.assertIn((changed_file, "head-sha"), io_impl.load_json_calls)
        self.assertEqual([("airflow", "a.py")], io_impl.resolve_calls)
        self.assertEqual([("sources/airflow/a.py", "head-sha")], io_impl.load_text_calls)

    def test_build_rule_diffs_caches_source_loads_per_ref_and_path(self) -> None:
        changed_file = (
            "private/its-enterprise/ruling/src/test/resources/expected_ruling/"
            "airflow/python-S107.json"
        )
        io_impl = FakeRulingDiffIO(
            json_by_ref_path={
                (changed_file, "base-sha"): {"airflow:a.py": [1]},
                (changed_file, "head-sha"): {"airflow:a.py": [2, 2]},
            },
            text_by_ref_path={
                ("sources/airflow/a.py", "base-sha"): "base\ncontent\n",
                ("sources/airflow/a.py", "head-sha"): "head\ncontent\n",
            },
        )

        core.build_rule_diffs([changed_file], "base-sha", "head-sha", io_impl)

        self.assertEqual(1, io_impl.load_text_calls.count(("sources/airflow/a.py", "base-sha")))
        self.assertEqual(1, io_impl.load_text_calls.count(("sources/airflow/a.py", "head-sha")))

    def test_build_rule_diffs_missing_source_produces_placeholder_snippet(self) -> None:
        changed_file = (
            "private/its-enterprise/ruling/src/test/resources/expected_ruling/"
            "airflow/python-S107.json"
        )
        io_impl = FakeRulingDiffIO(
            json_by_ref_path={
                (changed_file, "base-sha"): {"airflow:a.py": [1]},
                (changed_file, "head-sha"): {"airflow:a.py": [1, 3]},
            },
            text_by_ref_path={("sources/airflow/a.py", "head-sha"): None},
        )

        diffs = core.build_rule_diffs([changed_file], "base-sha", "head-sha", io_impl)

        self.assertEqual(1, len(diffs[0].snippets))
        self.assertEqual(
            "(source file not found at this revision: a.py)",
            diffs[0].snippets[0].body,
        )


class SourceLoadingTest(unittest.TestCase):
    @patch("ruling_diff_io.subprocess.run")
    def test_load_text_at_ref_reads_from_submodule_commit(self, mocked_run) -> None:
        mocked_run.side_effect = [
            subprocess.CompletedProcess(
                args=["git", "rev-parse"],
                returncode=0,
                stdout="subsha123\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "-C", "sources", "show"],
                returncode=0,
                stdout="print('from submodule')\n",
                stderr="",
            ),
        ]

        content = io.load_text_at_ref(
            "private/its-enterprise/sources_ruling/project/foo.py", "deadbeef"
        )

        self.assertEqual("print('from submodule')\n", content)

    @patch("ruling_diff_io.subprocess.run")
    def test_load_text_at_ref_falls_back_to_workspace_copy_with_warning(
        self, mocked_run
    ) -> None:
        mocked_run.return_value = subprocess.CompletedProcess(
            args=["git"],
            returncode=128,
            stdout="",
            stderr="fatal: path 'private/its-enterprise/sources_ruling/foo.py' exists on disk, but not in 'deadbeef'",
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write("print('from workspace')\n")
            tmp_path = tmp.name
        try:
            with self.assertLogs(level="WARNING") as logs:
                content = io.load_text_at_ref(tmp_path, "deadbeef")
            self.assertEqual("print('from workspace')\n", content)
            self.assertTrue(any("using workspace copy" in log for log in logs.output))
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)

    @patch("ruling_diff_io.subprocess.run")
    def test_load_text_at_ref_warns_when_source_missing(self, mocked_run) -> None:
        mocked_run.return_value = subprocess.CompletedProcess(
            args=["git"],
            returncode=128,
            stdout="",
            stderr="fatal: path 'missing.py' exists on disk, but not in 'deadbeef'",
        )
        with self.assertLogs(level="WARNING") as logs:
            content = io.load_text_at_ref("missing.py", "deadbeef")
        self.assertIsNone(content)
        self.assertTrue(any("not found at deadbeef" in log for log in logs.output))


class GitHubCommentLookupTest(unittest.TestCase):
    @patch("ruling_diff_io.run_command")
    def test_get_existing_comment_id_reads_all_pages(self, mocked_run_command) -> None:
        mocked_run_command.return_value = (
            '[{"id": 1, "body": "first"}]\n'
            '[{"id": 2, "body": "text <!-- ruling-diff-comment -->"}]\n'
        )

        comment_id = io.get_existing_comment_id(
            "895", "SonarSource/sonar-python-enterprise"
        )

        self.assertEqual("2", comment_id)

    def test_parse_json_documents_handles_multiple_arrays(self) -> None:
        documents = io.parse_json_documents('[{"a":1}]\n[{"b":2}]')
        self.assertEqual(2, len(documents))


class GitHubActionIOTest(unittest.TestCase):
    def test_resolve_source_path_for_project_rulings_uses_path_directly(self) -> None:
        io_impl = io.GitHubActionIO()
        self.assertEqual(
            "private/its-enterprise/sources_ruling/biopython/Bio/Nexus/Nexus.py",
            io_impl.resolve_source_path("project", "biopython/Bio/Nexus/Nexus.py"),
        )

    def test_resolve_source_path_for_project_rulings_falls_back_to_sources_child(self) -> None:
        io_impl = io.GitHubActionIO()
        self.assertEqual(
            "private/its-enterprise/sources_ruling/specific-rules/S1716.py",
            io_impl.resolve_source_path("project", "S1716.py"),
        )

    def test_resolve_source_path_for_project_rulings_falls_back_to_sources_internal(
        self,
    ) -> None:
        io_impl = io.GitHubActionIO()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sources_ruling = f"{tmp_dir}/sources_ruling"
            sources_internal = f"{tmp_dir}/sources_internal_ruling"
            sources_namespace = f"{tmp_dir}/sources_internal_namespace_ruling"
            pathlib.Path(sources_ruling).mkdir(parents=True, exist_ok=True)
            pathlib.Path(sources_internal).mkdir(parents=True, exist_ok=True)
            pathlib.Path(sources_namespace).mkdir(parents=True, exist_ok=True)
            target = f"{sources_internal}/foo.py"
            pathlib.Path(target).write_text("x\n", encoding="utf-8")
            with (
                patch.object(io, "RULING_SOURCES_SUBMODULE", sources_ruling),
                patch.object(io, "SOURCES_INTERNAL_RULING_ROOT", sources_internal),
                patch.object(
                    io,
                    "SOURCES_INTERNAL_NAMESPACE_RULING_ROOT",
                    sources_namespace,
                ),
            ):
                self.assertEqual(target, io_impl.resolve_source_path("project", "foo.py"))

    def test_resolve_source_path_for_project_rulings_falls_back_to_namespace_child(
        self,
    ) -> None:
        io_impl = io.GitHubActionIO()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sources_ruling = f"{tmp_dir}/sources_ruling"
            sources_internal = f"{tmp_dir}/sources_internal_ruling"
            sources_namespace = f"{tmp_dir}/sources_internal_namespace_ruling"
            namespace_child = f"{sources_namespace}/basic_namespace"
            pathlib.Path(sources_ruling).mkdir(parents=True, exist_ok=True)
            pathlib.Path(sources_internal).mkdir(parents=True, exist_ok=True)
            pathlib.Path(namespace_child).mkdir(parents=True, exist_ok=True)
            target = f"{namespace_child}/foo.py"
            pathlib.Path(target).write_text("x\n", encoding="utf-8")
            with (
                patch.object(io, "RULING_SOURCES_SUBMODULE", sources_ruling),
                patch.object(io, "SOURCES_INTERNAL_RULING_ROOT", sources_internal),
                patch.object(
                    io,
                    "SOURCES_INTERNAL_NAMESPACE_RULING_ROOT",
                    sources_namespace,
                ),
            ):
                self.assertEqual(target, io_impl.resolve_source_path("project", "foo.py"))

    def test_resolve_source_path_for_project_rulings_returns_primary_on_miss(self) -> None:
        io_impl = io.GitHubActionIO()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sources_ruling = f"{tmp_dir}/sources_ruling"
            sources_internal = f"{tmp_dir}/sources_internal_ruling"
            sources_namespace = f"{tmp_dir}/sources_internal_namespace_ruling"
            pathlib.Path(sources_ruling).mkdir(parents=True, exist_ok=True)
            pathlib.Path(sources_internal).mkdir(parents=True, exist_ok=True)
            pathlib.Path(sources_namespace).mkdir(parents=True, exist_ok=True)
            primary = f"{sources_ruling}/missing.py"
            with (
                patch.object(io, "RULING_SOURCES_SUBMODULE", sources_ruling),
                patch.object(io, "SOURCES_INTERNAL_RULING_ROOT", sources_internal),
                patch.object(
                    io,
                    "SOURCES_INTERNAL_NAMESPACE_RULING_ROOT",
                    sources_namespace,
                ),
            ):
                self.assertEqual(primary, io_impl.resolve_source_path("project", "missing.py"))

    def test_resolve_source_path_uses_project_overrides(self) -> None:
        io_impl = io.GitHubActionIO()
        self.assertEqual(
            "private/its-enterprise/sources_ruling/mypy-0.782/pkg/file.py",
            io_impl.resolve_source_path("mypy", "pkg/file.py"),
        )

    def test_resolve_source_path_uses_default_project_root(self) -> None:
        io_impl = io.GitHubActionIO()
        self.assertEqual(
            "private/its-enterprise/sources_ruling/custom-project/pkg/file.py",
            io_impl.resolve_source_path("custom-project", "/pkg/file.py"),
        )


if __name__ == "__main__":
    unittest.main()
