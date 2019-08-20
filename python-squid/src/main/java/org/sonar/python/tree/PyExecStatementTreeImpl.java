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
import com.sonar.sslr.api.Token;
import org.sonar.python.api.tree.PyExecStatementTree;
import org.sonar.python.api.tree.PyExpressionTree;

public class PyExecStatementTreeImpl extends PyTree implements PyExecStatementTree {
  private final Token execKeyword;
  private final PyExpressionTree expression;
  private final PyExpressionTree globalsExpression;
  private final PyExpressionTree localsExpression;

  public PyExecStatementTreeImpl(AstNode astNode, Token execKeyword, PyExpressionTree expression, PyExpressionTree globalsExpression, PyExpressionTree localsExpression) {
    super(astNode);
    this.execKeyword = execKeyword;
    this.expression = expression;
    this.globalsExpression = globalsExpression;
    this.localsExpression = localsExpression;
  }

  public PyExecStatementTreeImpl(AstNode astNode, Token execKeyword, PyExpressionTree expression) {
    super(astNode);
    this.execKeyword = execKeyword;
    this.expression = expression;
    globalsExpression = null;
    localsExpression = null;
  }

  @Override
  public Token execKeyword() {
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
}
