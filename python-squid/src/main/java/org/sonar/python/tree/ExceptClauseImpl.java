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

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.ExceptClause;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.StatementList;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.Tree;

public class ExceptClauseImpl extends PyTree implements ExceptClause {
  private final Token exceptKeyword;
  private final StatementList body;
  private final Expression exception;
  private final Token asKeyword;
  private final Token commaToken;
  private final Expression exceptionInstance;

  public ExceptClauseImpl(Token exceptKeyword, StatementList body) {
    super(exceptKeyword, body.lastToken());
    this.exceptKeyword = exceptKeyword;
    this.body = body;
    this.exception = null;
    this.asKeyword = null;
    this.commaToken = null;
    this.exceptionInstance = null;
  }

  public ExceptClauseImpl(Token exceptKeyword, StatementList body,
                          Expression exception, @Nullable Token asNode, @Nullable Token commaNode, Expression exceptionInstance) {
    super(exceptKeyword, body.lastToken());
    this.exceptKeyword = exceptKeyword;
    this.body = body;
    this.exception = exception;
    this.asKeyword = asNode;
    this.commaToken = commaNode;
    this.exceptionInstance = exceptionInstance;
  }

  public ExceptClauseImpl(Token exceptKeyword, StatementList body, Expression exception) {
    super(exceptKeyword, body.lastToken());
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
  public StatementList body() {
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
  public Expression exception() {
    return exception;
  }

  @CheckForNull
  @Override
  public Expression exceptionInstance() {
    return exceptionInstance;
  }

  @Override
  public Kind getKind() {
    return Kind.EXCEPT_CLAUSE;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitExceptClause(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(exceptKeyword, exception, asKeyword, exceptionInstance, commaToken, body).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
