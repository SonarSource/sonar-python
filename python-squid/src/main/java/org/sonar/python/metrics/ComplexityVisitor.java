/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.metrics;

import com.sonar.sslr.api.TokenType;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.tree.PyBinaryExpressionTree;
import org.sonar.python.api.tree.PyComprehensionIfTree;
import org.sonar.python.api.tree.PyConditionalExpressionTree;
import org.sonar.python.api.tree.PyForStatementTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyIfStatementTree;
import org.sonar.python.api.tree.PyWhileStatementTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.tree.BaseTreeVisitor;

public class ComplexityVisitor extends BaseTreeVisitor {

  private int complexity = 0;

  public static int complexity(Tree pyTree) {
    ComplexityVisitor visitor = pyTree.is(Tree.Kind.FUNCDEF) ? new FunctionComplexityVisitor() : new ComplexityVisitor();
    pyTree.accept(visitor);
    return visitor.complexity;
  }

  @Override
  public void visitFunctionDef(PyFunctionDefTree pyFunctionDefTree) {
    complexity++;
    super.visitFunctionDef(pyFunctionDefTree);
  }

  @Override
  public void visitForStatement(PyForStatementTree pyForStatementTree) {
    complexity++;
    super.visitForStatement(pyForStatementTree);
  }

  @Override
  public void visitWhileStatement(PyWhileStatementTree pyWhileStatementTree) {
    complexity++;
    super.visitWhileStatement(pyWhileStatementTree);
  }

  @Override
  public void visitIfStatement(PyIfStatementTree pyIfStatementTree) {
    if (!pyIfStatementTree.isElif()) {
      complexity++;
    }
    super.visitIfStatement(pyIfStatementTree);
  }

  @Override
  public void visitConditionalExpression(PyConditionalExpressionTree pyConditionalExpressionTree) {
    complexity++;
    super.visitConditionalExpression(pyConditionalExpressionTree);
  }

  @Override
  public void visitBinaryExpression(PyBinaryExpressionTree pyBinaryExpressionTree) {
    TokenType type = pyBinaryExpressionTree.operator().type();
    if (type.equals(PythonKeyword.AND) || type.equals(PythonKeyword.OR)) {
      complexity++;
    }
    super.visitBinaryExpression(pyBinaryExpressionTree);
  }

  @Override
  public void visitComprehensionIf(PyComprehensionIfTree tree) {
    complexity++;
    super.visitComprehensionIf(tree);
  }

  public int getComplexity() {
    return complexity;
  }

  private static class FunctionComplexityVisitor extends ComplexityVisitor {

    private int functionNestingLevel = 0;

    @Override
    public void visitFunctionDef(PyFunctionDefTree pyFunctionDefTree) {
      functionNestingLevel++;
      if (functionNestingLevel == 1) {
        super.visitFunctionDef(pyFunctionDefTree);
      }
      functionNestingLevel--;
    }
  }
}
