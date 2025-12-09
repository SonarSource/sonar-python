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

import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;

public class TestFileTelemetryCollector {

  private static final Set<String> TEST_FRAMEWORK_MODULES = Set.of("unittest", "pytest");

  private final AtomicLong totalMainFiles = new AtomicLong(0);
  private final AtomicLong misclassifiedTestFiles = new AtomicLong(0);

  public void collect(FileInput rootTree, InputFile.Type fileType) {
    if (fileType != InputFile.Type.MAIN) {
      return;
    }

    totalMainFiles.incrementAndGet();

    var importVisitor = new TestImportVisitor();
    rootTree.accept(importVisitor);

    if (importVisitor.hasTestFrameworkImport) {
      misclassifiedTestFiles.incrementAndGet();
      return;
    }

    var pytestPatternVisitor = new PytestPatternVisitor();
    rootTree.accept(pytestPatternVisitor);

    if (pytestPatternVisitor.hasPytestPattern) {
      misclassifiedTestFiles.incrementAndGet();
    }
  }

  public TestFileTelemetry getTelemetry() {
    return new TestFileTelemetry(totalMainFiles.get(), misclassifiedTestFiles.get());
  }

  private static class TestImportVisitor extends BaseTreeVisitor {
    boolean hasTestFrameworkImport = false;

    @Override
    public void visitImportName(ImportName importName) {
      if (hasTestFrameworkImport) {
        return;
      }

      for (var aliasedName : importName.modules()) {
        var names = aliasedName.dottedName().names();
        if (!names.isEmpty()) {
          String moduleName = names.get(0).name();
          if (TEST_FRAMEWORK_MODULES.contains(moduleName)) {
            hasTestFrameworkImport = true;
            return;
          }
        }
      }
      super.visitImportName(importName);
    }

    @Override
    public void visitImportFrom(ImportFrom importFrom) {
      if (hasTestFrameworkImport) {
        return;
      }

      var moduleName = importFrom.module();
      if (moduleName != null) {
        var names = moduleName.names();
        if (!names.isEmpty()) {
          String rootModule = names.get(0).name();
          if (TEST_FRAMEWORK_MODULES.contains(rootModule)) {
            hasTestFrameworkImport = true;
            return;
          }
        }
      }
      super.visitImportFrom(importFrom);
    }
  }

  private static class PytestPatternVisitor extends BaseTreeVisitor {
    boolean hasPytestPattern = false;

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      if (hasPytestPattern) {
        return;
      }

      String functionName = functionDef.name().name();
      if (functionName.startsWith("test_") && containsAssert(functionDef)) {
        hasPytestPattern = true;
        return;
      }
      super.visitFunctionDef(functionDef);
    }

    private static boolean containsAssert(FunctionDef functionDef) {
      var assertVisitor = new AssertStatementVisitor();
      functionDef.body().accept(assertVisitor);
      return assertVisitor.hasAssert;
    }
  }

  private static class AssertStatementVisitor extends BaseTreeVisitor {
    boolean hasAssert = false;

    @Override
    public void visitAssertStatement(AssertStatement assertStatement) {
      hasAssert = true;
    }
  }
}

