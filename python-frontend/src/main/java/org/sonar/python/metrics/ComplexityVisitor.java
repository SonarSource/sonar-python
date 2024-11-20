/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.metrics;

import com.sonar.sslr.api.TokenType;
import org.sonar.python.api.PythonKeyword;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.ComprehensionIf;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;

public class ComplexityVisitor extends BaseTreeVisitor {

  private int complexity = 0;

  public static int complexity(Tree pyTree) {
    ComplexityVisitor visitor = pyTree.is(Tree.Kind.FUNCDEF) ? new FunctionComplexityVisitor() : new ComplexityVisitor();
    pyTree.accept(visitor);
    return visitor.complexity;
  }

  @Override
  public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
    complexity++;
    super.visitFunctionDef(pyFunctionDefTree);
  }

  @Override
  public void visitForStatement(ForStatement pyForStatementTree) {
    complexity++;
    super.visitForStatement(pyForStatementTree);
  }

  @Override
  public void visitWhileStatement(WhileStatement pyWhileStatementTree) {
    complexity++;
    super.visitWhileStatement(pyWhileStatementTree);
  }

  @Override
  public void visitIfStatement(IfStatement pyIfStatementTree) {
    if (!pyIfStatementTree.isElif()) {
      complexity++;
    }
    super.visitIfStatement(pyIfStatementTree);
  }

  @Override
  public void visitConditionalExpression(ConditionalExpression pyConditionalExpressionTree) {
    complexity++;
    super.visitConditionalExpression(pyConditionalExpressionTree);
  }

  @Override
  public void visitBinaryExpression(BinaryExpression pyBinaryExpressionTree) {
    TokenType type = pyBinaryExpressionTree.operator().type();
    if (type.equals(PythonKeyword.AND) || type.equals(PythonKeyword.OR)) {
      complexity++;
    }
    super.visitBinaryExpression(pyBinaryExpressionTree);
  }

  @Override
  public void visitComprehensionIf(ComprehensionIf tree) {
    complexity++;
    super.visitComprehensionIf(tree);
  }

  public int getComplexity() {
    return complexity;
  }

  private static class FunctionComplexityVisitor extends ComplexityVisitor {

    private int functionNestingLevel = 0;

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      functionNestingLevel++;
      if (functionNestingLevel == 1) {
        super.visitFunctionDef(pyFunctionDefTree);
      }
      functionNestingLevel--;
    }
  }
}
