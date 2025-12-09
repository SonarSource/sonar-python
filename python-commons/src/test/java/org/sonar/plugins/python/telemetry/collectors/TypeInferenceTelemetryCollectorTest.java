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

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

import static org.assertj.core.api.Assertions.assertThat;

class TypeInferenceTelemetryCollectorTest {

  private static PythonVisitorContext createContext(String code) {
    var mockFile = new TestPythonVisitorRunner.MockPythonFile("", "test.py", code);
    return TestPythonVisitorRunner.createContext(mockFile, null, "", ProjectLevelSymbolTable.empty(), CacheContextImpl.dummyCache());
  }

  @Test
  void collectsNamesWithKnownTypes() {
    var context = createContext("""
      x = 1
      y = "hello"
      z = x + 2
      """);

    var collector = new TypeInferenceTelemetryCollector();
    collector.collect(context.rootTree());

    var telemetry = collector.getTelemetry();
    // Names with symbols: x (assignment), y (assignment), z (assignment), x (usage)
    assertThat(telemetry.totalNames()).isEqualTo(4);
    // Unique symbols: x, y, z
    assertThat(telemetry.uniqueSymbols()).isEqualTo(3);
    assertThat(telemetry.unknownTypeNames()).isZero();
    assertThat(telemetry.unresolvedImportTypeNames()).isZero();
    assertThat(telemetry.totalImports()).isZero();
    assertThat(telemetry.importsWithUnknownType()).isZero();
  }

  @Test
  void collectsUnknownTypes() {
    var context = createContext("""
      x = foo(1)
      """);

    var collector = new TypeInferenceTelemetryCollector();
    collector.collect(context.rootTree());

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalNames()).isEqualTo(2);
    assertThat(telemetry.uniqueSymbols()).isEqualTo(1); // only x
    assertThat(telemetry.unknownTypeNames()).isEqualTo(2);
  }

  @Test
  void collectComplexAssignment() {
    var context = createContext("""
      obj = object()
      obj.attr = 1
      """);

    var collector = new TypeInferenceTelemetryCollector();
    collector.collect(context.rootTree());

    var telemetry = collector.getTelemetry();
    assertThat(telemetry.totalNames()).isEqualTo(4);
    assertThat(telemetry.uniqueSymbols()).isEqualTo(1);
    assertThat(telemetry.unknownTypeNames()).isEqualTo(1);
    assertThat(telemetry.totalImports()).isZero();
    assertThat(telemetry.importsWithUnknownType()).isZero();
  }

  @Test
  void collectsUnresolvedImportTypes() {
    var context = createContext("""
      import unknown_module
      from another_unknown import something

      x = unknown_module.foo
      y = something.bar
      """);

    var collector = new TypeInferenceTelemetryCollector();
    collector.collect(context.rootTree());

    var telemetry = collector.getTelemetry();
    // Names with symbols: unknown_module (import), something (import), x (assignment), unknown_module (usage),
    // foo (attribute), y (assignment), something (usage), bar (attribute), and dotted name parts
    assertThat(telemetry.totalNames()).isEqualTo(9);
    // Unique symbols: unknown_module, something, x, y
    assertThat(telemetry.uniqueSymbols()).isEqualTo(4);
    // Two imports: import unknown_module and from another_unknown import something
    assertThat(telemetry.totalImports()).isEqualTo(2);
    assertThat(telemetry.importsWithUnknownType()).isEqualTo(2);
    // unknown_module and something have UnresolvedImportType
    assertThat(telemetry.unresolvedImportTypeNames()).isGreaterThanOrEqualTo(2);
  }

  @Test
  void aggregatesAcrossMultipleFiles() {
    var context1 = createContext("""
      x = 1
      y = 2
      """);

    var context2 = createContext("""
      a = "hello"
      b = "world"
      """);

    var collector = new TypeInferenceTelemetryCollector();
    collector.collect(context1.rootTree());
    collector.collect(context2.rootTree());

    var telemetry = collector.getTelemetry();
    // File 1: x (assignment), y (assignment) = 2 names; File 2: a (assignment), b (assignment) = 2 names; Total = 4
    assertThat(telemetry.totalNames()).isEqualTo(4);
    // Unique symbols: x, y, a, b = 4
    assertThat(telemetry.uniqueSymbols()).isEqualTo(4);
    assertThat(telemetry.unknownTypeNames()).isZero();
    assertThat(telemetry.unresolvedImportTypeNames()).isZero();
    assertThat(telemetry.totalImports()).isZero();
    assertThat(telemetry.importsWithUnknownType()).isZero();
  }


  @Test
  void typeInferenceTelemetryRecordOperations() {
    var empty = TypeInferenceTelemetry.empty();
    assertThat(empty.totalNames()).isZero();
    assertThat(empty.unknownTypeNames()).isZero();
    assertThat(empty.unresolvedImportTypeNames()).isZero();
    assertThat(empty.totalImports()).isZero();
    assertThat(empty.importsWithUnknownType()).isZero();
    assertThat(empty.uniqueSymbols()).isZero();
    assertThat(empty.unknownSymbols()).isZero();

    var telemetry1 = new TypeInferenceTelemetry(10, 2, 1, 5, 3, 5, 1);
    var telemetry2 = new TypeInferenceTelemetry(5, 1, 2, 3, 1, 3, 2);

    var combined = telemetry1.add(telemetry2);
    assertThat(combined.totalNames()).isEqualTo(15);
    assertThat(combined.unknownTypeNames()).isEqualTo(3);
    assertThat(combined.unresolvedImportTypeNames()).isEqualTo(3);
    assertThat(combined.totalImports()).isEqualTo(8);
    assertThat(combined.importsWithUnknownType()).isEqualTo(4);
    assertThat(combined.uniqueSymbols()).isEqualTo(8);
    assertThat(combined.unknownSymbols()).isEqualTo(3);
  }

  @Test
  void countsImportFromStatements() {
    var context = createContext("""
      from unknown_lib import func1, func2 # unknown types
      from another_lib import func3 as f3 # unknown type
      from typing import List # known type
      """);

    var collector = new TypeInferenceTelemetryCollector();
    collector.collect(context.rootTree());

    var telemetry = collector.getTelemetry();
    // Total of 4 imported items: func1, func2, func3, List
    assertThat(telemetry.totalImports()).isEqualTo(4);
    // 3 have unknown types: func1, func2, func3
    assertThat(telemetry.importsWithUnknownType()).isEqualTo(3);
  }

  @Test
  void countsImportNameStatements() {
    var context = createContext("""
      import unknown_lib # unknown type
      import another_lib as al # unknown type
      import typing # known type
      """);

    var collector = new TypeInferenceTelemetryCollector();
    collector.collect(context.rootTree());

    var telemetry = collector.getTelemetry();
    // Total of 3 imports: unknown_lib, another_lib, typing
    assertThat(telemetry.totalImports()).isEqualTo(3);
    // 2 have unknown types: unknown_lib, another_lib
    assertThat(telemetry.importsWithUnknownType()).isEqualTo(2);
  }
}
