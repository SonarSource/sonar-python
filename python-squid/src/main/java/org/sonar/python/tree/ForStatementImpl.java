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
import java.util.Collections;
import java.util.Objects;
import org.sonar.python.api.tree.Token;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.ForStatement;
import org.sonar.python.api.tree.StatementList;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.Tree;

public class ForStatementImpl extends PyTree implements ForStatement {

  private final Token forKeyword;
  private final List<Expression> expressions;
  private final Token inKeyword;
  private final List<Expression> testExpressions;
  private final Token colon;
  private final StatementList body;
  @Nullable
  private final Token elseKeyword;
  @Nullable
  private final Token elseColon;
  private final StatementList elseBody;
  private final Token asyncKeyword;
  private final boolean isAsync;

  public ForStatementImpl(AstNode astNode, Token forKeyword, List<Expression> expressions, Token inKeyword,
                          List<Expression> testExpressions, Token colon, StatementList body,
                          @Nullable Token elseKeyword, @Nullable Token elseColon, @Nullable StatementList elseBody, @Nullable Token asyncKeyword) {
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
  public void accept(TreeVisitor visitor) {
    visitor.visitForStatement(this);
  }

  @Override
  public Token forKeyword() {
    return forKeyword;
  }

  @Override
  public List<Expression> expressions() {
    return expressions;
  }

  @Override
  public Token inKeyword() {
    return inKeyword;
  }

  @Override
  public List<Expression> testExpressions() {
    return testExpressions;
  }

  @Override
  public Token colon() {
    return colon;
  }

  @Override
  public StatementList body() {
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
  public StatementList elseBody() {
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
    return Stream.of(Arrays.asList(asyncKeyword, forKeyword), expressions, Collections.singletonList(inKeyword), testExpressions,
      Arrays.asList(colon, body, elseKeyword, colon, elseBody)).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
