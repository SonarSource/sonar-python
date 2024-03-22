/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.Tree;

public class ExceptClauseImpl extends PyTree implements ExceptClause {
  private final Token exceptKeyword;
  private final Token starToken;
  private final Token colon;
  private final Token newLine;
  private final Token indent;
  private final StatementList body;
  private final Token dedent;
  private final Expression exception;
  private final Token asKeyword;
  private final Token commaToken;
  private final Expression exceptionInstance;

  public ExceptClauseImpl(Token exceptKeyword, @Nullable Token starToken, Token colon, @Nullable Token newLine, @Nullable Token indent, StatementList body,
                          @Nullable Token dedent) {
    this.exceptKeyword = exceptKeyword;
    this.starToken = starToken;
    this.colon = colon;
    this.newLine = newLine;
    this.indent = indent;
    this.body = body;
    this.dedent = dedent;
    this.exception = null;
    this.asKeyword = null;
    this.commaToken = null;
    this.exceptionInstance = null;
  }

  public ExceptClauseImpl(Token exceptKeyword, @Nullable Token starToken, Token colon, @Nullable Token newLine, @Nullable Token indent, StatementList body,
                          @Nullable Token dedent, Expression exception, @Nullable Token asNode, @Nullable Token commaNode, Expression exceptionInstance) {
    this.exceptKeyword = exceptKeyword;
    this.starToken = starToken;
    this.colon = colon;
    this.newLine = newLine;
    this.indent = indent;
    this.body = body;
    this.dedent = dedent;
    this.exception = exception;
    this.asKeyword = asNode;
    this.commaToken = commaNode;
    this.exceptionInstance = exceptionInstance;
  }

  public ExceptClauseImpl(Token exceptKeyword, @Nullable Token starToken, Token colon, @Nullable Token newLine, @Nullable Token indent, StatementList body,
                          @Nullable Token dedent, Expression exception) {
    this.exceptKeyword = exceptKeyword;
    this.starToken = starToken;
    this.colon = colon;
    this.newLine = newLine;
    this.indent = indent;
    this.body = body;
    this.dedent = dedent;
    this.exception = exception;
    this.asKeyword = null;
    this.commaToken = null;
    this.exceptionInstance = null;
  }

  @Override
  public Token exceptKeyword() {
    return exceptKeyword;
  }

  @CheckForNull
  @Override
  public Token starToken() {
    return starToken;
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
  public Token colon() {
    return colon;
  }

  @Override
  public Kind getKind() {
    return this.starToken != null ? Kind.EXCEPT_GROUP_CLAUSE : Kind.EXCEPT_CLAUSE;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitExceptClause(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(exceptKeyword, starToken, exception, asKeyword, commaToken, exceptionInstance, colon, newLine, indent, body, dedent)
      .filter(Objects::nonNull).toList();
  }
}
