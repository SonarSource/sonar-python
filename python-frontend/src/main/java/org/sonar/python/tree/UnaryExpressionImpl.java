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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.HasTypeDependencies;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.v2.PythonType;

public class UnaryExpressionImpl extends PyTree implements UnaryExpression, HasTypeDependencies {

  private static final Map<String, Kind> KINDS_BY_OPERATOR = kindsByOperator();

  private final Kind kind;
  private final Token operator;
  private final Expression expression;
  private PythonType type = PythonType.UNKNOWN;

  private static Map<String, Kind> kindsByOperator() {
    Map<String, Kind> map = new HashMap<>();
    map.put("+", Kind.UNARY_PLUS);
    map.put("-", Kind.UNARY_MINUS);
    map.put("~", Kind.BITWISE_COMPLEMENT);
    map.put("not", Kind.NOT);
    return map;
  }

  public UnaryExpressionImpl(Token operator, Expression expression) {
    this.kind = KINDS_BY_OPERATOR.get(operator.value());
    this.operator = operator;
    this.expression = expression;
  }

  @Override
  public Token operator() {
    return operator;
  }

  @Override
  public Expression expression() {
    return expression;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitUnaryExpression(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(operator, expression).filter(Objects::nonNull).toList();
  }

  @Override
  public Kind getKind() {
    return kind;
  }

  @Override
  public InferredType type() {
    if (is(Kind.NOT)) {
      return InferredTypes.BOOL;
    }
    if (is(Kind.UNARY_MINUS, Kind.UNARY_PLUS)) {
      return expression.type();
    }
    return InferredTypes.anyType();
  }

  public UnaryExpression typeV2(PythonType type) {
    this.type = type;
    return this;
  }

  @Override
  public PythonType typeV2() {
    return type;
  }

  @Override
  public List<Expression> typeDependencies() {
    return List.of(expression);
  }
}
