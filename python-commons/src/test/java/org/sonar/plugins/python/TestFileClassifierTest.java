/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;

class TestFileClassifierTest {

  // --- looksLikeTestFileByPath ---

  @Test
  void empty_path_returns_false() {
    assertThat(TestFileClassifier.looksLikeTestFileByPath("")).isFalse();
  }

  @Test
  void file_in_tests_directory_returns_true() {
    assertThat(TestFileClassifier.looksLikeTestFileByPath("tests/foo.py")).isTrue();
  }

  @Test
  void file_in_test_singular_directory_returns_true() {
    assertThat(TestFileClassifier.looksLikeTestFileByPath("test/foo.py")).isTrue();
  }

  @Test
  void file_in_nested_tests_directory_returns_true() {
    assertThat(TestFileClassifier.looksLikeTestFileByPath("src/mypackage/tests/foo.py")).isTrue();
  }

  @Test
  void test_prefix_filename_returns_true() {
    assertThat(TestFileClassifier.looksLikeTestFileByPath("test_foo.py")).isTrue();
    assertThat(TestFileClassifier.looksLikeTestFileByPath("src/test_bar.py")).isTrue();
  }

  @Test
  void test_suffix_filename_returns_true() {
    assertThat(TestFileClassifier.looksLikeTestFileByPath("foo_test.py")).isTrue();
    assertThat(TestFileClassifier.looksLikeTestFileByPath("src/bar_test.py")).isTrue();
  }

  @Test
  void regular_file_in_src_returns_false() {
    assertThat(TestFileClassifier.looksLikeTestFileByPath("src/foo.py")).isFalse();
    assertThat(TestFileClassifier.looksLikeTestFileByPath("module.py")).isFalse();
  }

  @Test
  void directory_named_testing_not_matched() {
    // only "test" and "tests" trigger — "testing" must not
    assertThat(TestFileClassifier.looksLikeTestFileByPath("testing/foo.py")).isFalse();
  }

  @Test
  void windows_style_path_normalized() {
    assertThat(TestFileClassifier.looksLikeTestFileByPath("src\\tests\\foo.py")).isTrue();
  }

  // --- looksLikeTestFile with null tree ---

  @Test
  void null_tree_non_test_path_returns_false() {
    assertThat(TestFileClassifier.looksLikeTestFile("src/regular.py", null)).isFalse();
  }

  @Test
  void null_tree_test_path_returns_true() {
    assertThat(TestFileClassifier.looksLikeTestFile("tests/foo.py", null)).isTrue();
  }

  // --- import-based detection (visitImportName / visitImportFrom) ---

  @ParameterizedTest
  @ValueSource(strings = {
    "import unittest\n",
    "import pytest\n",
    "from unittest import TestCase\n",
    "from pytest import mark\n"
  })
  void test_framework_import_detected(String code) {
    FileInput tree = parse(code);
    assertThat(TestFileClassifier.looksLikeTestFile("regular.py", tree)).isTrue();
  }

  @Test
  void multiple_imports_early_exit_after_first_match() {
    // Second `import unittest` must hit the early-exit branch (hasTestFrameworkImport already true)
    FileInput tree = parse("import unittest\nimport unittest\n");
    assertThat(TestFileClassifier.looksLikeTestFile("regular.py", tree)).isTrue();
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "import os\nimport sys\n",              // non-test imports
    "from . import foo\n",                  // relative import — null module, must not throw
    "def test_something():\n    pass\n",    // test_ prefix but no assert
    "def do_something():\n    assert True\n" // assert but no test_ prefix
  })
  void ast_heuristic_not_detected(String code) {
    assertThat(TestFileClassifier.looksLikeTestFile("regular.py", parse(code))).isFalse();
  }

  @Test
  void multiple_from_imports_early_exit_after_first_match() {
    // Second from-import hits the early-exit branch (hasTestFrameworkImport already true)
    FileInput tree = parse("from unittest import TestCase\nfrom unittest import mock\n");
    assertThat(TestFileClassifier.looksLikeTestFile("regular.py", tree)).isTrue();
  }

  // --- pytest pattern detection (visitFunctionDef + AssertVisitor) ---

  @Test
  void test_function_with_assert_detected() {
    FileInput tree = parse("def test_something():\n    assert True\n");
    assertThat(TestFileClassifier.looksLikeTestFile("regular.py", tree)).isTrue();
  }

  @Test
  void multiple_test_functions_early_exit_after_first_match() {
    // Second test_ function hits the early-exit branch (hasPytestPattern already true)
    FileInput tree = parse("def test_one():\n    assert True\ndef test_two():\n    assert True\n");
    assertThat(TestFileClassifier.looksLikeTestFile("regular.py", tree)).isTrue();
  }

  @Test
  void path_match_short_circuits_ast_check() {
    // Path already matches — tree is never examined
    FileInput tree = parse("import os\n");
    assertThat(TestFileClassifier.looksLikeTestFile("tests/foo.py", tree)).isTrue();
  }

  // --- isImportBasedTestFile short-circuit (||) ---

  @Test
  void import_based_short_circuits_pytest_pattern_check() {
    // import unittest is true → isPytestPatternFile never called (|| short-circuit)
    FileInput tree = parse("import unittest\ndef regular_function():\n    pass\n");
    assertThat(TestFileClassifier.looksLikeTestFile("regular.py", tree)).isTrue();
  }

  private static FileInput parse(String code) {
    var astNode = PythonParser.create().parse(code);
    return new PythonTreeMaker().fileInput(astNode);
  }
}
