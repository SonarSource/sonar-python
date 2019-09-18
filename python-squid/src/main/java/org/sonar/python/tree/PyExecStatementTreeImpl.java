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
package org.sonar.python.tree;

import com.sonar.sslr.api.AstNode;
import org.sonar.python.api.tree.PyToken;
import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyExecStatementTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyExecStatementTreeImpl extends PyTree implements PyExecStatementTree {
  private final PyToken execKeyword;
  private final PyExpressionTree expression;
  private final PyExpressionTree globalsExpression;
  private final PyExpressionTree localsExpression;

  public PyExecStatementTreeImpl(AstNode astNode, PyToken execKeyword, PyExpressionTree expression,
                                 @Nullable PyExpressionTree globalsExpression, @Nullable PyExpressionTree localsExpression) {
    super(astNode);
    this.execKeyword = execKeyword;
    this.expression = expression;
    this.globalsExpression = globalsExpression;
    this.localsExpression = localsExpression;
  }

  public PyExecStatementTreeImpl(AstNode astNode, PyToken execKeyword, PyExpressionTree expression) {
    super(astNode);
    this.execKeyword = execKeyword;
    this.expression = expression;
    globalsExpression = null;
    localsExpression = null;
  }

  @Override
  public PyToken execKeyword() {
    return execKeyword;
  }

  @Override
  public PyExpressionTree expression() {
    return expression;
  }

  @Override
  public PyExpressionTree globalsExpression() {
    return globalsExpression;
  }

  @Override
  public PyExpressionTree localsExpression() {
    return localsExpression;
  }

  @Override
  public Kind getKind() {
    return Kind.EXEC_STMT;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitExecStatement(this);
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(expression, globalsExpression, localsExpression);
  }
}
