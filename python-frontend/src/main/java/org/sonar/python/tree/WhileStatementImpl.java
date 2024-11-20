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
package org.sonar.python.tree;

import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.WhileStatement;

public class WhileStatementImpl extends PyTree implements WhileStatement {

  private final Token whileKeyword;
  private final Expression condition;
  private final Token colon;
  private final Token firstNewline;
  private final Token firstIndent;
  private final StatementList body;
  private final Token firstDedent;
  private final ElseClause elseClause;

  public WhileStatementImpl(Token whileKeyword, Expression condition, Token colon, @Nullable Token firstNewline,
                            @Nullable Token firstIndent, StatementList body, @Nullable Token firstDedent, @Nullable ElseClause elseClause) {
    this.whileKeyword = whileKeyword;
    this.condition = condition;
    this.colon = colon;
    this.firstNewline = firstNewline;
    this.firstIndent = firstIndent;
    this.body = body;
    this.firstDedent = firstDedent;
    this.elseClause = elseClause;
  }

  @Override
  public Kind getKind() {
    return Kind.WHILE_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitWhileStatement(this);
  }

  @Override
  public Token whileKeyword() {
    return whileKeyword;
  }

  @Override
  public Expression condition() {
    return condition;
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
  public ElseClause elseClause() {
    return elseClause;
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(whileKeyword, condition, colon, firstNewline, firstIndent, body, firstDedent,
      elseClause).filter(Objects::nonNull)
      .toList();
  }
}
