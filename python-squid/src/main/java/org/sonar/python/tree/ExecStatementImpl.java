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
import javax.annotation.Nullable;
import org.sonar.python.api.tree.ExecStatement;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.TreeVisitor;

public class ExecStatementImpl extends PyTree implements ExecStatement {
  private final Token execKeyword;
  private final Expression expression;
  private final Expression globalsExpression;
  private final Expression localsExpression;
  private final Token separator;

  public ExecStatementImpl(Token execKeyword, Expression expression,
                                 @Nullable Expression globalsExpression, @Nullable Expression localsExpression, @Nullable Token separator) {
    super(execKeyword, localsExpression == null ? (globalsExpression == null ? expression.lastToken() : globalsExpression.lastToken()): localsExpression.lastToken());
    this.execKeyword = execKeyword;
    this.expression = expression;
    this.globalsExpression = globalsExpression;
    this.localsExpression = localsExpression;
    this.separator = separator;
  }

  public ExecStatementImpl(Token execKeyword, Expression expression, @Nullable Token separator) {
    super(execKeyword, expression.lastToken());
    this.execKeyword = execKeyword;
    this.expression = expression;
    globalsExpression = null;
    localsExpression = null;
    this.separator =separator;
  }

  @Override
  public Token execKeyword() {
    return execKeyword;
  }

  @Override
  public Expression expression() {
    return expression;
  }

  @Override
  public Expression globalsExpression() {
    return globalsExpression;
  }

  @Override
  public Expression localsExpression() {
    return localsExpression;
  }

  @Nullable
  @Override
  public Token separator() {
    return separator;
  }

  @Override
  public Kind getKind() {
    return Kind.EXEC_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitExecStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(execKeyword, expression, globalsExpression, localsExpression, separator).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
