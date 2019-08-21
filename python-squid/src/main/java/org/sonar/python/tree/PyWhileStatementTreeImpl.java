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
import java.util.List;
import javax.annotation.CheckForNull;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyStatementTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.PyWhileStatementTree;

public class PyWhileStatementTreeImpl extends PyTree implements PyWhileStatementTree {

  private final PyExpressionTree condition;
  private final List<PyStatementTree> body;
  private final List<PyStatementTree> elseBody;

  public PyWhileStatementTreeImpl(AstNode astNode, PyExpressionTree condition, List<PyStatementTree> body, List<PyStatementTree> elseBody) {
    super(astNode);
    this.condition = condition;
    this.body = body;
    this.elseBody = elseBody;
  }

  @Override
  public Kind getKind() {
    return Kind.WHILE_STMT;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitWhileStatement(this);
  }

  @Override
  public Token whileKeyword() {
    return null;
  }

  @Override
  public PyExpressionTree condition() {
    return condition;
  }

  @Override
  public Token colon() {
    return null;
  }

  @Override
  public List<PyStatementTree> body() {
    return body;
  }

  @CheckForNull
  @Override
  public Token elseKeyword() {
    return null;
  }

  @CheckForNull
  @Override
  public Token elseColon() {
    return null;
  }

  @CheckForNull
  @Override
  public List<PyStatementTree> elseBody() {
    return elseBody;
  }
}
