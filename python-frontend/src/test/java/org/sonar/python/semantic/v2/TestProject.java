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
package org.sonar.python.semantic.v2;

import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.TupleImpl;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeChecker;
import org.sonar.python.types.v2.TypesTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;
import static org.sonar.python.PythonTestUtils.pythonFile;

public class TestProject {
  private final ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
  private final ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);

  private static String toFileName(String modulePath) {
    int lastSlashIndex = modulePath.lastIndexOf("/");
    if (lastSlashIndex == -1) {
      // no slash -> the modulePath is the file name
      return modulePath;
    }
    return modulePath.substring(lastSlashIndex + 1);
  }

  private static String toPackageName(String modulePath) {
    int lastSlashIndex = modulePath.lastIndexOf("/");
    if (lastSlashIndex == -1) {
      return "";
    }
    return modulePath.substring(0, lastSlashIndex).replace("/", ".");
  }

  public TestProject addModule(String modulePath, String code) {
    String fileName = toFileName(modulePath);
    String packageName = toPackageName(modulePath);

    FileInput tree = parseWithoutSymbols(code);
    projectLevelSymbolTable.addModule(tree, packageName, pythonFile(fileName));
    return this;
  }

  public Expression lastExpression(String code) {
    return TypeInferenceV2Test.lastExpression(code, projectLevelTypeTable);
  }

  public TupleImpl lastExpressionAsTuple(String code) {
    Expression lastExpr = lastExpression(code);
    assertThat(lastExpr).isInstanceOf(TupleImpl.class);
    return (TupleImpl) lastExpr;
  }

  public FileInput inferTypes(String code) {
    return inferTypes("mod.py", code);
  }

  public FileInput inferTypes(String modulePath, String code) {
    String fileName = toFileName(modulePath);
    String packageName = toPackageName(modulePath);

    return TypesTestUtils.parseAndInferTypes(packageName, projectLevelTypeTable, pythonFile(fileName), code);
  }

  public TypeCheckBuilder typeCheckBuilder() {
    return new TypeChecker(projectLevelTypeTable).typeCheckBuilder();
  }

  public ProjectLevelTypeTable projectLevelTypeTable() {
    return projectLevelTypeTable;
  }
}
