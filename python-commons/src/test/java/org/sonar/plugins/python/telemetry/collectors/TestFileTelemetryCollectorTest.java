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

  private static PythonVisitorContext createContext(String code) {
    var mockFile = new TestPythonVisitorRunner.MockPythonFile("", "test.py", code);
    return TestPythonVisitorRunner.createContext(mockFile, null, "", ProjectLevelSymbolTable.empty(), CacheContextImpl.dummyCache());
  }

  @Test
  void lineCountForSingleLineFile() {
    var singleLine = createContext("x = 1");
    assertThat(TestFileTelemetryCollector.lineCount(singleLine.rootTree())).isEqualTo(1);
  }

  @Test
  void lineCountForMultiLineFile() {
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
    collector.collect(context.rootTree(), InputFile.Type.MAIN);

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isEqualTo(1);
    assertThat(telemetry.misclassifiedTestFiles()).isZero();
    assertThat(telemetry.totalLines()).isPositive();
    assertThat(telemetry.totalMainLines()).isEqualTo(telemetry.totalLines());
    assertThat(telemetry.testLines()).isZero();
    assertThat(telemetry.misclassifiedTestLines()).isZero();
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
    collector.collect(context.rootTree(), InputFile.Type.MAIN);

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isEqualTo(1);
    assertThat(telemetry.misclassifiedTestFiles()).isEqualTo(1);
    assertThat(telemetry.totalLines()).isPositive();
    assertThat(telemetry.totalMainLines()).isEqualTo(telemetry.totalLines());
    assertThat(telemetry.testLines()).isZero();
    assertThat(telemetry.misclassifiedTestLines()).isEqualTo(telemetry.totalMainLines());
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
    collector.collect(context.rootTree(), InputFile.Type.TEST);

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isZero();
    assertThat(telemetry.misclassifiedTestFiles()).isZero();
    assertThat(telemetry.totalLines()).isPositive();
    assertThat(telemetry.totalMainLines()).isZero();
    assertThat(telemetry.testLines()).isEqualTo(telemetry.totalLines());
    assertThat(telemetry.misclassifiedTestLines()).isZero();
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
  void aggregatesAcrossMultipleFiles() {
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
    collector.collect(context1.rootTree(), InputFile.Type.MAIN);
    collector.collect(context2.rootTree(), InputFile.Type.MAIN);
    collector.collect(context3.rootTree(), InputFile.Type.MAIN);

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isEqualTo(3);
    assertThat(telemetry.misclassifiedTestFiles()).isEqualTo(2);
    assertThat(telemetry.totalLines()).isEqualTo(6);
    assertThat(telemetry.totalMainLines()).isEqualTo(6);
    assertThat(telemetry.testLines()).isZero();
    assertThat(telemetry.misclassifiedTestLines()).isEqualTo(4);
  }

  @Test
  void mixedMainAndTestFiles() {
    var mainWithTest = createContext("import unittest");
    var mainWithoutTest = createContext("import os");
    var testFile = createContext("import pytest");

    var collector = new TestFileTelemetryCollector();
    collector.collect(mainWithTest.rootTree(), InputFile.Type.MAIN);
    collector.collect(mainWithoutTest.rootTree(), InputFile.Type.MAIN);
    collector.collect(testFile.rootTree(), InputFile.Type.TEST);

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isEqualTo(2);
    assertThat(telemetry.misclassifiedTestFiles()).isEqualTo(1);
    assertThat(telemetry.totalLines()).isEqualTo(3);
    assertThat(telemetry.totalMainLines()).isEqualTo(2);
    assertThat(telemetry.testLines()).isEqualTo(1);
    assertThat(telemetry.misclassifiedTestLines()).isEqualTo(1);
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
    collector.collect(context.rootTree(), InputFile.Type.MAIN);

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainFiles()).isEqualTo(1);
    assertThat(telemetry.misclassifiedTestFiles()).isEqualTo(1);
    assertThat(telemetry.totalLines()).isEqualTo(5);
    assertThat(telemetry.totalMainLines()).isEqualTo(5);
    assertThat(telemetry.testLines()).isZero();
    assertThat(telemetry.misclassifiedTestLines()).isEqualTo(5);
  }

  @Test
  void lineCountAggregatesCorrectlyForDifferentFileSizes() {
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
    collector.collect(smallFile.rootTree(), InputFile.Type.MAIN);
    collector.collect(largerFile.rootTree(), InputFile.Type.MAIN);

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainLines()).isEqualTo(11);
    assertThat(telemetry.misclassifiedTestLines()).isZero();
  }

  @Test
  void misclassifiedTestLinesOnlyCountsMisclassifiedFiles() {
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
    collector.collect(misclassifiedFile.rootTree(), InputFile.Type.MAIN);
    collector.collect(regularFile.rootTree(), InputFile.Type.MAIN);

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalMainLines()).isEqualTo(10);
    assertThat(telemetry.misclassifiedTestLines()).isEqualTo(5);
  }
}

