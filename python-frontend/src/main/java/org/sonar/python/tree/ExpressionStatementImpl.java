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

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class ExpressionStatementImpl extends SimpleStatement implements ExpressionStatement {
  private final List<Expression> expressions;
  private final Separators separators;

  public ExpressionStatementImpl(List<Expression> expressions, Separators separators) {
    this.expressions = expressions;
    this.separators = separators;
  }

  @Override
  public Kind getKind() {
    return Kind.EXPRESSION_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitExpressionStatement(this);
  }

  @Override
  public List<Expression> expressions() {
    return expressions;
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(expressions, separators.elements()).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Token separator() {
    return separators.last();
  }
}
