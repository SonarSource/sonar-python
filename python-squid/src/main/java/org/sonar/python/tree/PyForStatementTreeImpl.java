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
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyForStatementTree;
import org.sonar.python.api.tree.PyStatementListTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyForStatementTreeImpl extends PyTree implements PyForStatementTree {

  private final Token forKeyword;
  private final List<PyExpressionTree> expressions;
  private final Token inKeyword;
  private final List<PyExpressionTree> testExpressions;
  private final Token colon;
  private final PyStatementListTree body;
  @Nullable
  private final Token elseKeyword;
  @Nullable
  private final Token elseColon;
  private final PyStatementListTree elseBody;
  private final Token asyncKeyword;
  private final boolean isAsync;

  public PyForStatementTreeImpl(AstNode astNode, Token forKeyword, List<PyExpressionTree> expressions, Token inKeyword,
                                List<PyExpressionTree> testExpressions, Token colon, PyStatementListTree body, @Nullable Token elseKeyword,
                                @Nullable Token elseColon, @Nullable PyStatementListTree elseBody, Token asyncKeyword) {
    super(astNode);
    this.forKeyword = forKeyword;
    this.expressions = expressions;
    this.inKeyword = inKeyword;
    this.testExpressions = testExpressions;
    this.colon = colon;
    this.body = body;
    this.elseKeyword = elseKeyword;
    this.elseColon = elseColon;
    this.elseBody = elseBody;
    this.asyncKeyword = asyncKeyword;
    this.isAsync = asyncKeyword != null;
  }

  @Override
  public Kind getKind() {
    return Kind.FOR_STMT;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitForStatement(this);
  }

  @Override
  public Token forKeyword() {
    return forKeyword;
  }

  @Override
  public List<PyExpressionTree> expressions() {
    return expressions;
  }

  @Override
  public Token inKeyword() {
    return inKeyword;
  }

  @Override
  public List<PyExpressionTree> testExpressions() {
    return testExpressions;
  }

  @Override
  public Token colon() {
    return colon;
  }

  @Override
  public PyStatementListTree body() {
    return body;
  }

  @CheckForNull
  @Override
  public Token elseKeyword() {
    return elseKeyword;
  }

  @CheckForNull
  @Override
  public Token elseColon() {
    return elseColon;
  }

  @CheckForNull
  @Override
  public PyStatementListTree elseBody() {
    return elseBody;
  }

  @Override
  public boolean isAsync() {
    return isAsync;
  }

  @CheckForNull
  @Override
  public Token asyncKeyword() {
    return asyncKeyword;
  }

  @Override
  public List<Tree> children() {
    return Stream.of(expressions, testExpressions, Arrays.asList(body, elseBody))
      .flatMap(List::stream).collect(Collectors.toList());
  }
}
