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
import org.sonar.python.api.tree.PyExceptClauseTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyStatementTree;
import org.sonar.python.api.tree.PyTreeVisitor;

public class PyExceptClauseTreeImpl extends PyTree implements PyExceptClauseTree {
  private final Token exceptKeyword;
  private final List<PyStatementTree> body;
  private final PyExpressionTree exception;
  private final Token asKeyword;
  private final Token commaToken;
  private final PyExpressionTree exceptionInstance;

  public PyExceptClauseTreeImpl(AstNode astNode, Token exceptKeyword, List<PyStatementTree> body) {
    super(astNode);
    this.exceptKeyword = exceptKeyword;
    this.body = body;
    this.exception = null;
    this.asKeyword = null;
    this.commaToken = null;
    this.exceptionInstance = null;
  }

  public PyExceptClauseTreeImpl(AstNode astNode, Token exceptKeyword, List<PyStatementTree> body, PyExpressionTree exception, AstNode asNode, AstNode commaNode, PyExpressionTree exceptionInstance) {
    super(astNode);
    this.exceptKeyword = exceptKeyword;
    this.body = body;
    this.exception = exception;
    this.asKeyword = asNode != null ? asNode.getToken() : null;
    this.commaToken = commaNode != null ? commaNode.getToken() : null;
    this.exceptionInstance = exceptionInstance;
  }

  public PyExceptClauseTreeImpl(AstNode except, Token exceptKeyword, List<PyStatementTree> body, PyExpressionTree exception) {
    super(except);
    this.exceptKeyword = exceptKeyword;
    this.body = body;
    this.exception = exception;
    this.asKeyword = null;
    this.commaToken = null;
    this.exceptionInstance = null;
  }

  @Override
  public Token exceptKeyword() {
    return exceptKeyword;
  }

  @Override
  public List<PyStatementTree> body() {
    return body;
  }

  @CheckForNull
  @Override
  public Token asKeyword() {
    return asKeyword;
  }

  @CheckForNull
  @Override
  public Token commaToken() {
    return commaToken;
  }

  @CheckForNull
  @Override
  public PyExpressionTree exception() {
    return exception;
  }

  @CheckForNull
  @Override
  public PyExpressionTree exceptionInstance() {
    return exceptionInstance;
  }

  @Override
  public Kind getKind() {
    return Kind.EXCEPT_CLAUSE;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitExceptClause(this);
  }
}
