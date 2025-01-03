/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.semantic.v2.types;

import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.TryStatement;

public class TryStatementVisitor extends BaseTreeVisitor {
  // TODO: could be replaced with TreeUtils method call
  private boolean hasTryStatement = false;

  @Override
  public void visitClassDef(ClassDef classDef) {
    // Don't visit nested classes
  }

  @Override
  public void visitFunctionDef(FunctionDef visited) {
    // Don't visit nested functions
  }

  @Override
  public void visitTryStatement(TryStatement tryStatement) {
    hasTryStatement = true;
  }

  public boolean hasTryStatement() {
    return hasTryStatement;
  }
}
