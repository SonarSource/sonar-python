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
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.StatementList;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.WhileStatement;
import org.sonar.python.api.tree.Tree;

public class WhileStatementImpl extends PyTree implements WhileStatement {

  private final Token whileKeyword;
  private final Expression condition;
  private final Token colon;
  private final StatementList body;
  private final Token elseKeyword;
  private final Token elseColon;
  private final StatementList elseBody;

  public WhileStatementImpl(AstNode astNode, Token whileKeyword, Expression condition, Token colon, StatementList body,
                            @Nullable Token elseKeyword, @Nullable Token elseColon, @Nullable StatementList elseBody) {
    super(astNode);
    this.whileKeyword = whileKeyword;
    this.condition = condition;
    this.colon = colon;
    this.body = body;
    this.elseKeyword = elseKeyword;
    this.elseColon = elseColon;
    this.elseBody = elseBody;
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
  public List<Tree> children() {
    return Stream.of(whileKeyword, condition, colon, body, elseKeyword, elseColon, elseBody).filter(Objects::nonNull)
      .collect(Collectors.toList());
  }
}
