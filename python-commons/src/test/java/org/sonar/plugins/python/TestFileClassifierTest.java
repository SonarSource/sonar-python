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

import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.api.config.Configuration;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.sonarsource.analyzer.commons.appsec.TestFileClassifier.HEURISTIC_DISABLED_KEY;

class TestFileClassifierTest {

  @Test
  void test_source_configuration_disables_the_heuristic() {
    var config = mock(Configuration.class);
    when(config.get(anyString())).thenReturn(Optional.empty());
    when(config.getBoolean(anyString())).thenReturn(Optional.empty());

    assertThat(TestFileClassifier.isTestSourceConfigured(config)).isFalse();

    when(config.getBoolean("sonar.python.testFileHeuristic.disabled")).thenReturn(Optional.of(true));
    assertThat(TestFileClassifier.isTestSourceConfigured(config)).isTrue();

    when(config.getBoolean(HEURISTIC_DISABLED_KEY)).thenReturn(Optional.of(true));
    assertThat(TestFileClassifier.isTestSourceConfigured(config)).isTrue();
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "tests/helpers.py",
    "src/TESTING/factories.py",
    "src\\test\\conftest.py",
    "src/TestCase.py",
    "src/latest.py"
  })
  void path_signals_include_test_support_files_and_aggressive_filename_matches(String path) {
    assertThat(TestFileClassifier.looksLikeTestFileByPath(path)).isTrue();
  }

  @ParameterizedTest
  @ValueSource(strings = {"", "src/production.py", "src/module.py"})
  void non_test_paths_are_not_classified_without_an_ast_signal(String path) {
    assertThat(TestFileClassifier.looksLikeTestFileByPath(path)).isFalse();
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "import pytest as pt\n",
    "from unittest.mock import Mock\n",
    "import doctest\n",
    "from django.test import TestCase\n",
    "import hypothesis.strategies as st\n",
    "import robot\n",
    "import behave\n",
    "import pytest_bdd\n",
    "import nose\n",
    "import nose2\n",
    "from twisted.trial import unittest\n",
    "import testtools\n",
    "import mock\n",
    "import pytest_mock\n",
    "import fixtures\n",
    "import factory\n",
    "import factory_boy\n",
    "import freezegun\n",
    "import responses\n",
    "import requests_mock\n",
    "import respx\n",
    "import httpretty\n",
    "import moto\n",
    "import vcr\n",
    "import testcontainers\n"
  })
  void test_framework_and_utility_imports_are_detected(String code) {
    assertThat(TestFileClassifier.looksLikeTestFile("src/production.py", parse(code))).isTrue();
  }

  @Test
  void test_shaped_top_level_declarations_are_detected() {
    assertThat(TestFileClassifier.looksLikeTestFile("src/production.py", parse("class TestService:\n    pass\n"))).isTrue();
    assertThat(TestFileClassifier.looksLikeTestFile("src/production.py", parse("""
      async def TEST_first():
          pass
      def test_second():
          pass
      def helper():
          pass
      """))).isTrue();
  }

  @Test
  void function_ratio_requires_a_strict_majority() {
    assertThat(TestFileClassifier.looksLikeTestFile("src/production.py", parse("""
      def test_first():
          pass
      def helper():
          pass
      """))).isFalse();
    assertThat(TestFileClassifier.looksLikeTestFile("src/production.py", parse("pass\n"))).isFalse();
  }

  @Test
  void nested_test_signals_do_not_affect_top_level_classification() {
    assertThat(TestFileClassifier.looksLikeTestFile("src/production.py", parse("""
      def helper():
          import pytest
          def test_inner():
              pass
      """))).isFalse();
  }

  @Test
  void path_signal_is_available_when_parsing_fails() {
    assertThat(TestFileClassifier.looksLikeTestFile("tests/helpers.py", null)).isTrue();
    assertThat(TestFileClassifier.looksLikeTestFile("src/production.py", null)).isFalse();
  }

  private static FileInput parse(String code) {
    var astNode = PythonParser.create().parse(code);
    return new PythonTreeMaker().fileInput(astNode);
  }
}
