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
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.HasTypeDependencies;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.v2.PythonType;

import static org.sonar.python.types.InferredTypes.DECL_INT;
import static org.sonar.python.types.InferredTypes.DECL_STR;
import static org.sonar.python.types.InferredTypes.INT;
import static org.sonar.python.types.InferredTypes.STR;

public class BinaryExpressionImpl extends PyTree implements BinaryExpression, HasTypeDependencies {

  private static final Map<String, Kind> KINDS_BY_OPERATOR = Map.ofEntries(
    Map.entry("+", Kind.PLUS),
    Map.entry("-", Kind.MINUS),
    Map.entry("*", Kind.MULTIPLICATION),
    Map.entry("/", Kind.DIVISION),
    Map.entry("//", Kind.FLOOR_DIVISION),
    Map.entry("%", Kind.MODULO),
    Map.entry("**", Kind.POWER),
    Map.entry("@", Kind.MATRIX_MULTIPLICATION),
    Map.entry(">>", Kind.SHIFT_EXPR),
    Map.entry("<<", Kind.SHIFT_EXPR),
    Map.entry("&", Kind.BITWISE_AND),
    Map.entry("|", Kind.BITWISE_OR),
    Map.entry("^", Kind.BITWISE_XOR),
    Map.entry("and", Kind.AND),
    Map.entry("or", Kind.OR),
    Map.entry("==", Kind.COMPARISON),
    Map.entry("<=", Kind.COMPARISON),
    Map.entry(">=", Kind.COMPARISON),
    Map.entry("<", Kind.COMPARISON),
    Map.entry(">", Kind.COMPARISON),
    Map.entry("!=", Kind.COMPARISON),
    Map.entry("<>", Kind.COMPARISON),
    Map.entry("in", Kind.IN),
    Map.entry("is", Kind.IS)
  );

  private final Kind kind;
  private final Expression leftOperand;
  private final Token operator;
  private final Expression rightOperand;
  private PythonType type = PythonType.UNKNOWN;

  public BinaryExpressionImpl(Expression leftOperand, Token operator, Expression rightOperand) {
    this.kind = KINDS_BY_OPERATOR.get(operator.value());
    this.leftOperand = leftOperand;
    this.operator = operator;
    this.rightOperand = rightOperand;
  }

  @Override
  public Expression leftOperand() {
    return leftOperand;
  }

  @Override
  public Token operator() {
    return operator;
  }

  @Override
  public Expression rightOperand() {
    return rightOperand;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitBinaryExpression(this);
  }

  @Override
  public Kind getKind() {
    return kind;
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(leftOperand, operator, rightOperand).filter(Objects::nonNull).toList();
  }

  @Override
  public InferredType type() {
    if (is(Kind.AND, Kind.OR)) {
      return InferredTypes.or(leftOperand.type(), rightOperand.type());
    }
    if (is(Kind.PLUS)) {
      InferredType leftType = leftOperand.type();
      InferredType rightType = rightOperand.type();
      if (leftType.equals(INT) && rightType.equals(INT)) {
        return INT;
      }
      if (leftType.equals(STR) && rightType.equals(STR)) {
        return STR;
      }
      if (shouldReturnDeclaredInt(leftType, rightType)) {
        return DECL_INT;
      }
      if (shouldReturnDeclaredStr(leftType, rightType)) {
        return DECL_STR;
      }
    }
    return InferredTypes.anyType();
  }

  @Override
  public PythonType typeV2() {
    return type;
  }

  public BinaryExpressionImpl typeV2(PythonType type) {
    this.type = type;
    return this;
  }

  private static boolean shouldReturnDeclaredStr(InferredType leftType, InferredType rightType) {
    return (leftType.equals(DECL_STR) && rightType.equals(DECL_STR)) ||
      (leftType.equals(STR) && rightType.equals(DECL_STR)) ||
      (leftType.equals(DECL_STR) && rightType.equals(STR));
  }

  private static boolean shouldReturnDeclaredInt(InferredType leftType, InferredType rightType) {
    return (leftType.equals(DECL_INT) && rightType.equals(DECL_INT)) ||
      (leftType.equals(INT) && rightType.equals(DECL_INT)) ||
      (leftType.equals(DECL_INT) && rightType.equals(INT));
  }

  @Override
  public List<Expression> typeDependencies() {
    if (is(Kind.AND, Kind.OR, Kind.PLUS)) {
      return Arrays.asList(leftOperand, rightOperand);
    }
    return Collections.emptyList();
  }
}
