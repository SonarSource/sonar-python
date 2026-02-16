/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.telemetry.collectors;

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

import static org.assertj.core.api.Assertions.assertThat;

class TestFileTelemetryCollectorTest {

  private static final String NON_TEST_PATH = "src/main/module.py";

  private static PythonVisitorContext createContext(String code) {
    var mockFile = new TestPythonVisitorRunner.MockPythonFile("", "test.py", code);
    return TestPythonVisitorRunner.createContext(mockFile, null, "", ProjectLevelSymbolTable.empty(), CacheContextImpl.dummyCache());
  }

  @Test
  void lineCount_singleLineFile() {
    var singleLine = createContext("x = 1");
    assertThat(TestFileTelemetryCollector.lineCount(singleLine.rootTree())).isEqualTo(1);
  }

  @Test
  void lineCount_multiLineFile() {
    var multiLine = createContext("""
      x = 1
      y = 2
      z = 3
      """);
    assertThat(TestFileTelemetryCollector.lineCount(multiLine.rootTree())).isEqualTo(4);
  }

  @ParameterizedTest
  @MethodSource("provideNotMisclassifiedTestCases")
  void mainFile_notMisclassified(String code) {
    var context = createContext(code);

    var collector = new TestFileTelemetryCollector();
    collector.collect(context.rootTree(), InputFile.Type.MAIN, NON_TEST_PATH);

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isEqualTo(1);
    assertThat(telemetry.importBasedMisclassifiedTestFiles()).isZero();
    assertThat(telemetry.totalLines()).isPositive();
    assertThat(telemetry.totalMainLines()).isEqualTo(telemetry.totalLines());
    assertThat(telemetry.testLines()).isZero();
    assertThat(telemetry.importBasedMisclassifiedTestLines()).isZero();
    assertThat(telemetry.pathBasedMisclassifiedTestFiles()).isZero();
    assertThat(telemetry.pathBasedMisclassifiedTestLines()).isZero();
  }

  private static Stream<Arguments> provideNotMisclassifiedTestCases() {
    return Stream.of(
      Arguments.of("""
        import os
        import sys
        x = 1
        """),
      Arguments.of("""
        def test_something():
          pass
        """),
      Arguments.of("""
        def validate_something():
          pass
        """));
  }

  @ParameterizedTest
  @MethodSource("provideMisclassifiedTestCases")
  void mainFileWithTestImports_isMisclassified(String code) {
    var context = createContext(code);

    var collector = new TestFileTelemetryCollector();
    collector.collect(context.rootTree(), InputFile.Type.MAIN, NON_TEST_PATH);

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isEqualTo(1);
    assertThat(telemetry.importBasedMisclassifiedTestFiles()).isEqualTo(1);
    assertThat(telemetry.totalLines()).isPositive();
    assertThat(telemetry.totalMainLines()).isEqualTo(telemetry.totalLines());
    assertThat(telemetry.testLines()).isZero();
    assertThat(telemetry.importBasedMisclassifiedTestLines()).isEqualTo(telemetry.totalMainLines());
    assertThat(telemetry.pathBasedMisclassifiedTestFiles()).isZero();
    assertThat(telemetry.pathBasedMisclassifiedTestLines()).isZero();
  }

  private static Stream<Arguments> provideMisclassifiedTestCases() {
    return Stream.of(
      Arguments.of("""
        import unittest
        
        class MyTest(unittest.TestCase):
          pass
        """),
      Arguments.of("""
        import pytest
        
        def test_something():
          assert True
        """),
      Arguments.of("""
        import pytest.fixture
        import os
        
        @pytest.fixture
        def my_fixture():
          return 42
        """),
      Arguments.of("""
        from unittest import TestCase
        from os import path
        
        class MyTest(TestCase):
          pass
        """),
      Arguments.of("""
        from pytest import fixture
        
        @fixture
        def my_fixture():
          return 42
        """),
      Arguments.of("""
        from unittest.mock import Mock
        """)
    );
  }

  @ParameterizedTest
  @MethodSource("provideTestFileNotCountedCases")
  void testFile_countsLinesButNotMainMetrics(String code) {
    var context = createContext(code);

    var collector = new TestFileTelemetryCollector();
    collector.collect(context.rootTree(), InputFile.Type.TEST, "tests/test_module.py");

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isZero();
    assertThat(telemetry.importBasedMisclassifiedTestFiles()).isZero();
    assertThat(telemetry.totalLines()).isPositive();
    assertThat(telemetry.totalMainLines()).isZero();
    assertThat(telemetry.testLines()).isEqualTo(telemetry.totalLines());
    assertThat(telemetry.importBasedMisclassifiedTestLines()).isZero();
    assertThat(telemetry.pathBasedMisclassifiedTestFiles()).isZero();
    assertThat(telemetry.pathBasedMisclassifiedTestLines()).isZero();
  }

  private static Stream<Arguments> provideTestFileNotCountedCases() {
    return Stream.of(
      Arguments.of("""
        import unittest

        class MyTest(unittest.TestCase):
          pass
        """),
      Arguments.of("""
        from . import unittest
        from ... import other

        class MyTest(unittest.TestCase):
          pass
          """),
      Arguments.of("""
        import pytest

        def test_function():
          assert True
        """),
      Arguments.of("""
        import os
        def helper():
          pass
        """),
      Arguments.of("""
        from unittest.mock import Mock
        """),
      Arguments.of("""
        def test_another():
          assert 1 == 1
        """));
  }

  @Test
  void importBasedMisclassification_aggregatesAcrossMultipleFiles() {
    var context1 = createContext("""
      import unittest
      """);
    var context2 = createContext("""
      import pytest
      """);
    var context3 = createContext("""
      import os
      """);

    var collector = new TestFileTelemetryCollector();
    collector.collect(context1.rootTree(), InputFile.Type.MAIN, "src/main/module1.py");
    collector.collect(context2.rootTree(), InputFile.Type.MAIN, "src/main/module2.py");
    collector.collect(context3.rootTree(), InputFile.Type.MAIN, "src/main/module3.py");

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isEqualTo(3);
    assertThat(telemetry.importBasedMisclassifiedTestFiles()).isEqualTo(2);
    assertThat(telemetry.totalLines()).isEqualTo(6);
    assertThat(telemetry.totalMainLines()).isEqualTo(6);
    assertThat(telemetry.testLines()).isZero();
    assertThat(telemetry.importBasedMisclassifiedTestLines()).isEqualTo(4);
    assertThat(telemetry.pathBasedMisclassifiedTestFiles()).isZero();
    assertThat(telemetry.pathBasedMisclassifiedTestLines()).isZero();
  }

  @Test
  void mixedMainAndTestFiles_countsCorrectly() {
    var mainWithTest = createContext("import unittest");
    var mainWithoutTest = createContext("import os");
    var testFile = createContext("import pytest");

    var collector = new TestFileTelemetryCollector();
    collector.collect(mainWithTest.rootTree(), InputFile.Type.MAIN, "src/main/module1.py");
    collector.collect(mainWithoutTest.rootTree(), InputFile.Type.MAIN, "src/main/module2.py");
    collector.collect(testFile.rootTree(), InputFile.Type.TEST, "tests/test_module.py");

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isEqualTo(2);
    assertThat(telemetry.importBasedMisclassifiedTestFiles()).isEqualTo(1);
    assertThat(telemetry.totalLines()).isEqualTo(3);
    assertThat(telemetry.totalMainLines()).isEqualTo(2);
    assertThat(telemetry.testLines()).isEqualTo(1);
    assertThat(telemetry.importBasedMisclassifiedTestLines()).isEqualTo(1);
    assertThat(telemetry.pathBasedMisclassifiedTestFiles()).isZero();
    assertThat(telemetry.pathBasedMisclassifiedTestLines()).isZero();
  }

  @Test
  void mainFileWithTestFunction_isMisclassified() {
    var context = createContext("""
      def test_something():
        assert True
      def foo():
        pass
      """);

    var collector = new TestFileTelemetryCollector();
    collector.collect(context.rootTree(), InputFile.Type.MAIN, NON_TEST_PATH);

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isEqualTo(1);
    assertThat(telemetry.importBasedMisclassifiedTestFiles()).isEqualTo(1);
    assertThat(telemetry.totalLines()).isEqualTo(5);
    assertThat(telemetry.totalMainLines()).isEqualTo(5);
    assertThat(telemetry.testLines()).isZero();
    assertThat(telemetry.importBasedMisclassifiedTestLines()).isEqualTo(5);
    assertThat(telemetry.pathBasedMisclassifiedTestFiles()).isZero();
    assertThat(telemetry.pathBasedMisclassifiedTestLines()).isZero();
  }

  @Test
  void lineCount_aggregatesForDifferentFileSizes() {
    var smallFile = createContext("x = 1");
    var largerFile = createContext("""
      import os
      import sys
      import json
      
      def foo():
        pass
      
      def bar():
        pass
      """);

    var collector = new TestFileTelemetryCollector();
    collector.collect(smallFile.rootTree(), InputFile.Type.MAIN, "src/main/small.py");
    collector.collect(largerFile.rootTree(), InputFile.Type.MAIN, "src/main/large.py");

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainLines()).isEqualTo(11);
    assertThat(telemetry.importBasedMisclassifiedTestLines()).isZero();
    assertThat(telemetry.pathBasedMisclassifiedTestLines()).isZero();
  }

  @Test
  void misclassifiedTestLines_onlyCountMisclassifiedFiles() {
    var misclassifiedFile = createContext("""
      import pytest
      
      def test_something():
        assert True
      """);
    var regularFile = createContext("""
      import os
      
      def helper():
        pass
      """);

    var collector = new TestFileTelemetryCollector();
    collector.collect(misclassifiedFile.rootTree(), InputFile.Type.MAIN, "src/main/module1.py");
    collector.collect(regularFile.rootTree(), InputFile.Type.MAIN, "src/main/module2.py");

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainLines()).isEqualTo(10);
    assertThat(telemetry.importBasedMisclassifiedTestLines()).isEqualTo(5);
    assertThat(telemetry.pathBasedMisclassifiedTestLines()).isZero();
  }

  @Test
  void pathBasedMisclassifiedTestLines_onlyCountMisclassifiedFiles() {
    var misclassifiedFile = createContext("""
      import os

      def helper():
        pass
      """);
    var regularFile = createContext("""
      import os

      def other():
        pass
      """);

    var collector = new TestFileTelemetryCollector();
    collector.collect(misclassifiedFile.rootTree(), InputFile.Type.MAIN, "tests/module1.py");
    collector.collect(regularFile.rootTree(), InputFile.Type.MAIN, "src/main/module2.py");

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainLines()).isEqualTo(10);
    assertThat(telemetry.pathBasedMisclassifiedTestLines()).isEqualTo(5);
    assertThat(telemetry.importBasedMisclassifiedTestLines()).isZero();
  }

  @Test
  void pathBasedHeuristic_detectsTestDirectory() {
    var context = createContext("import os");

    var collector = new TestFileTelemetryCollector();
    collector.collect(context.rootTree(), InputFile.Type.MAIN, "tests/test_module.py");

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isEqualTo(1);
    assertThat(telemetry.importBasedMisclassifiedTestFiles()).isZero();
    assertThat(telemetry.pathBasedMisclassifiedTestFiles()).isEqualTo(1);
    assertThat(telemetry.pathBasedMisclassifiedTestLines()).isEqualTo(1);
    assertThat(telemetry.filesInPathBasedOnly()).isEqualTo(1);
    assertThat(telemetry.linesInPathBasedOnly()).isEqualTo(1);
  }

  @ParameterizedTest
  @MethodSource("providePathBasedHeuristicCases")
  void pathBasedHeuristic_classifiesCorrectly(String filePath, boolean expected) {
    assertThat(TestFileTelemetryCollector.isPathBasedMisclassifiedTestFile(filePath)).isEqualTo(expected);
  }

  private static Stream<Arguments> providePathBasedHeuristicCases() {
    return Stream.of(
      Arguments.of("src/tests/module.py", true),
      Arguments.of("src/Test/module.py", true),
      Arguments.of("C:\\project\\tests\\module.py", true),
      Arguments.of("src/package/tests/unit/module.py", true),
      Arguments.of("src/main/test_module.py", false),
      Arguments.of("src/testing/module.py", false),
      Arguments.of("", false)
    );
  }

  @Test
  void bothHeuristics_detectSameFile() {
    var context = createContext("import unittest");

    var collector = new TestFileTelemetryCollector();
    collector.collect(context.rootTree(), InputFile.Type.MAIN, "tests/test_module.py");

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.importBasedMisclassifiedTestFiles()).isEqualTo(1);
    assertThat(telemetry.pathBasedMisclassifiedTestFiles()).isEqualTo(1);
    assertThat(telemetry.filesInImportBasedOnly()).isZero();
    assertThat(telemetry.filesInPathBasedOnly()).isZero();
    assertThat(telemetry.linesInImportBasedOnly()).isZero();
    assertThat(telemetry.linesInPathBasedOnly()).isZero();
  }

  @Test
  void importBasedOnly_detectsFile() {
    var context = createContext("import pytest");

    var collector = new TestFileTelemetryCollector();
    collector.collect(context.rootTree(), InputFile.Type.MAIN, NON_TEST_PATH);

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.importBasedMisclassifiedTestFiles()).isEqualTo(1);
    assertThat(telemetry.pathBasedMisclassifiedTestFiles()).isZero();
    assertThat(telemetry.filesInImportBasedOnly()).isEqualTo(1);
    assertThat(telemetry.filesInPathBasedOnly()).isZero();
    assertThat(telemetry.linesInImportBasedOnly()).isEqualTo(1);
    assertThat(telemetry.linesInPathBasedOnly()).isZero();
  }

  @Test
  void pathBasedOnly_detectsFile() {
    var context = createContext("import os");

    var collector = new TestFileTelemetryCollector();
    collector.collect(context.rootTree(), InputFile.Type.MAIN, "tests/helper.py");

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.importBasedMisclassifiedTestFiles()).isZero();
    assertThat(telemetry.pathBasedMisclassifiedTestFiles()).isEqualTo(1);
    assertThat(telemetry.filesInImportBasedOnly()).isZero();
    assertThat(telemetry.filesInPathBasedOnly()).isEqualTo(1);
    assertThat(telemetry.linesInImportBasedOnly()).isZero();
    assertThat(telemetry.linesInPathBasedOnly()).isEqualTo(1);
  }

  @Test
  void heuristicComparison_aggregatesAcrossMultipleFiles() {
    var importBased = createContext("import unittest");
    var pathBased = createContext("import os");
    var both = createContext("import pytest");
    var neither = createContext("x = 1");

    var collector = new TestFileTelemetryCollector();
    collector.collect(importBased.rootTree(), InputFile.Type.MAIN, "src/main/module1.py");
    collector.collect(pathBased.rootTree(), InputFile.Type.MAIN, "tests/helper.py");
    collector.collect(both.rootTree(), InputFile.Type.MAIN, "tests/test_module.py");
    collector.collect(neither.rootTree(), InputFile.Type.MAIN, "src/main/module2.py");

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isEqualTo(4);
    assertThat(telemetry.importBasedMisclassifiedTestFiles()).isEqualTo(2);
    assertThat(telemetry.pathBasedMisclassifiedTestFiles()).isEqualTo(2);
    assertThat(telemetry.filesInImportBasedOnly()).isEqualTo(1);
    assertThat(telemetry.filesInPathBasedOnly()).isEqualTo(1); 
    assertThat(telemetry.linesInImportBasedOnly()).isEqualTo(1);
    assertThat(telemetry.linesInPathBasedOnly()).isEqualTo(1);
  }

}

