/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.ExecStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class ExecStatementImpl extends SimpleStatement implements ExecStatement {
  private final Token execKeyword;
  private final Expression expression;
  private final Token in;
  private final Expression globalsExpression;
  private final Token comma;
  private final Expression localsExpression;
  private final Separators separators;

  public ExecStatementImpl(Token execKeyword, Expression expression, Token in, @Nullable Expression globalsExpression,
                           @Nullable Token comma, @Nullable Expression localsExpression, Separators separators) {
    this.execKeyword = execKeyword;
    this.expression = expression;
    this.in = in;
    this.globalsExpression = globalsExpression;
    this.comma = comma;
    this.localsExpression = localsExpression;
    this.separators = separators;
  }

  public ExecStatementImpl(Token execKeyword, Expression expression, Separators separators) {
    this.execKeyword = execKeyword;
    this.expression = expression;
    this.in = null;
    this.globalsExpression = null;
    this.comma = null;
    this.localsExpression = null;
    this.separators = separators;
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
    return separators.last();
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
    return Stream.of(Arrays.asList(execKeyword, expression, in, globalsExpression, comma, localsExpression), separators.elements())
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
