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

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.HasTypeDependencies;
import org.sonar.python.types.v2.PythonType;

public class ParenthesizedExpressionImpl extends PyTree implements ParenthesizedExpression, HasTypeDependencies {

  private final Token leftParenthesis;
  private final Expression expression;
  private final Token rightParenthesis;

  public ParenthesizedExpressionImpl(Token leftParenthesis, Expression expression, Token rightParenthesis) {
    this.leftParenthesis = leftParenthesis;
    this.expression = expression;
    this.rightParenthesis = rightParenthesis;
  }

  @Override
  public Token leftParenthesis() {
    return leftParenthesis;
  }

  @Override
  public Expression expression() {
    return expression;
  }

  @Override
  public Token rightParenthesis() {
    return rightParenthesis;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitParenthesizedExpression(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(leftParenthesis, expression, rightParenthesis).toList();
  }

  @Override
  public Kind getKind() {
    return Kind.PARENTHESIZED;
  }

  @Override
  public InferredType type() {
    return expression.type();
  }

  @Override
  public PythonType typeV2() {
    return expression.typeV2();
  }

  @Override
  public List<Expression> typeDependencies() {
    return Collections.singletonList(expression);
  }
}
