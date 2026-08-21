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

import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.config.Configuration;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PythonTestFileClassifierTest {

  @ParameterizedTest
  @ValueSource(strings = {
    "tests/helpers.py",
    "src/TESTING/factories.py",
    "src/Test/helpers.py",
    "src/Tests/helpers.py",
    "src/Testing/factories.py",
    "src/test/conftest.py",
    "conftest.py",
    "src/test_module.py",
    "test_module.py",
    "src/testExample.py",
    "src/Test.py",
    "src/TestModule.py",
    "src/module_test.py",
    "module_test.py",
    "src/module_tests.py",
    "module_tests.py",
    "src/module_doctest.py",
    "module_doctest.py",
    "src/module_doctests.py",
    "module_doctests.py"
  })
  void path_signals_include_test_support_files_and_boundary_aware_filenames(String path) {
    assertThat(classifier().looksLikeTestFile(file(path), null)).isTrue();
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "",
    "src/production.py",
    "src/module.py",
    "src/latest/module.py",
    "src/greatest.py",
    "src/latest_adopted_release_filter.py",
    "src/primetest.py",
    "src/stubtest.py",
    "src/souptest.py",
    "src/send_test_email.py",
    "src/testExample.txt",
    "src/testexample.py",
    "src/Testing.py",
    "src/Testimonial.py"
  })
  void non_test_paths_are_not_classified_without_an_ast_signal(String path) {
    assertThat(classifier().looksLikeTestFile(file(path), null)).isFalse();
  }

  @Test
  void shared_classifier_disables_the_python_detector_when_test_sources_are_configured() {
    var configuration = config();
    when(configuration.get("sonar.tests")).thenReturn(Optional.of("tests"));

    assertThat(new PythonTestFileClassifier(configuration)
      .looksLikeTestFile(file("src/production.py"), parse("import pytest\n"))).isFalse();
    assertThat(new PythonTestFileClassifier(configuration)
      .looksLikeTestFile(file("src/TestModule.py"), null)).isFalse();
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "import pytest as pt\n",
    "from unittest.mock import Mock\n",
    "import doctest\n",
    "from django.test import TestCase\n",
    "from django import test\n",
    "from twisted import trial\n",
    "import hypothesis.strategies as st\n",
    "import robot\n",
    "import behave\n",
    "import pytest_bdd\n",
    "import nose\n",
    "import nose2\n",
    "from twisted.trial import unittest\n",
    "import testtools\n",
    "import pytest_mock\n",
    "import factory_boy\n",
    "import freezegun\n",
    "import requests_mock\n",
    "import respx\n",
    "import httpretty\n",
    "import moto\n",
    "import testcontainers\n"
  })
  void test_framework_and_utility_imports_are_detected(String code) {
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse(code))).isTrue();
  }

  @Test
  void non_test_submodule_imports_are_not_classified() {
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse("from django import forms\n"))).isFalse();
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse("from twisted import internet\n"))).isFalse();
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "import mock\n",
    "import fixtures\n",
    "import factory\n",
    "import responses\n",
    "import vcr\n"
  })
  void generic_utility_imports_alone_are_not_classified(String code) {
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse(code))).isFalse();
  }

  @Test
  void test_shaped_top_level_declarations_require_a_majority_of_two_or_more() {
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse("class TestService:\n    pass\n"))).isFalse();
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse("def test():\n    pass\n"))).isFalse();
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse("""
      class TestService:
          pass
      def test_synchronous():
          pass
      async def test_asynchronous():
          pass
      class ProductionService:
          pass
      """))).isTrue();
  }

  @Test
  void test_shaped_declarations_are_counted_with_other_top_level_classes() {
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse("""
      class Formatter:
          pass
      class HtmlFormatter:
          pass
      class TextFormatter:
          pass
      class JsonFormatter:
          pass
      class XmlFormatter:
          pass
      def test():
          pass
      """))).isFalse();
  }

  @Test
  void lowercase_continuations_after_test_are_not_test_shaped() {
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse("class Testing:\n    pass\n"))).isFalse();
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse("""
      def testing():
          pass
      def testimony():
          pass
      def helper():
          pass
      """))).isFalse();
  }

  @Test
  void test_shaped_declaration_ratio_requires_a_strict_majority() {
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse("""
      def test_first():
          pass
      def helper():
          pass
      """))).isFalse();
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse("""
      def test_first():
          pass
      def test_second():
          pass
      def first_helper():
          pass
      def second_helper():
          pass
      """))).isFalse();
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse("pass\n"))).isFalse();
  }

  @Test
  void nested_test_signals_do_not_affect_top_level_classification() {
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse("""
      def helper():
          import pytest
          def test_inner():
              pass
      """))).isFalse();
  }

  @Test
  void path_signal_is_available_when_parsing_fails() {
    assertThat(classifier().looksLikeTestFile(file("tests/helpers.py"), null)).isTrue();
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), null)).isFalse();
  }

  @Test
  void absent_or_relative_import_input_does_not_indicate_a_test_file() {
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), parse("from . import pytest\n"))).isFalse();
  }

  @Test
  void missing_ast_nodes_do_not_classify_a_file_as_a_test() {
    var fileInput = mock(FileInput.class);
    assertThat(classifier().looksLikeTestFile(file("src/production.py"), fileInput)).isFalse();

    var aliasedName = mock(AliasedName.class);
    var importName = mock(ImportName.class);
    var statements = mock(StatementList.class);
    when(importName.modules()).thenReturn(List.of(aliasedName));
    when(statements.statements()).thenReturn(List.of(importName));
    when(fileInput.statements()).thenReturn(statements);

    assertThat(classifier().looksLikeTestFile(file("src/production.py"), fileInput)).isFalse();
  }

  private static PythonTestFileClassifier classifier() {
    return new PythonTestFileClassifier(config());
  }

  private static Configuration config() {
    var configuration = mock(Configuration.class);
    when(configuration.get(anyString())).thenReturn(Optional.empty());
    when(configuration.getBoolean(anyString())).thenReturn(Optional.empty());
    return configuration;
  }

  private static InputFile file(String path) {
    var inputFile = mock(InputFile.class);
    when(inputFile.relativePath()).thenReturn(path);
    return inputFile;
  }

  private static FileInput parse(String code) {
    var astNode = PythonParser.create().parse(code);
    return new PythonTreeMaker().fileInput(astNode);
  }
}
