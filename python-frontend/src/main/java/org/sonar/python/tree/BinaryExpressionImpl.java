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

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
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
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeSource;
import org.sonar.python.types.v2.UnionType;

import static org.sonar.python.types.InferredTypes.DECL_INT;
import static org.sonar.python.types.InferredTypes.DECL_STR;
import static org.sonar.python.types.InferredTypes.INT;
import static org.sonar.python.types.InferredTypes.STR;

public class BinaryExpressionImpl extends PyTree implements BinaryExpression, HasTypeDependencies {

  private static final Map<String, Kind> KINDS_BY_OPERATOR = kindsByOperator();

  private final Kind kind;
  private final Expression leftOperand;
  private final Token operator;
  private final Expression rightOperand;

  private static Map<String, Kind> kindsByOperator() {
    Map<String, Kind> map = new HashMap<>();
    map.put("+", Kind.PLUS);
    map.put("-", Kind.MINUS);
    map.put("*", Kind.MULTIPLICATION);
    map.put("/", Kind.DIVISION);
    map.put("//", Kind.FLOOR_DIVISION);
    map.put("%", Kind.MODULO);
    map.put("**", Kind.POWER);
    map.put("@", Kind.MATRIX_MULTIPLICATION);
    map.put(">>", Kind.SHIFT_EXPR);
    map.put("<<", Kind.SHIFT_EXPR);
    map.put("&", Kind.BITWISE_AND);
    map.put("|", Kind.BITWISE_OR);
    map.put("^", Kind.BITWISE_XOR);
    map.put("and", Kind.AND);
    map.put("or", Kind.OR);
    map.put("==", Kind.COMPARISON);
    map.put("<=", Kind.COMPARISON);
    map.put(">=", Kind.COMPARISON);
    map.put("<", Kind.COMPARISON);
    map.put(">", Kind.COMPARISON);
    map.put("!=", Kind.COMPARISON);
    map.put("<>", Kind.COMPARISON);
    return map;
  }

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
    if (is(Kind.AND, Kind.OR)) {
      return UnionType.or(leftOperand.typeV2(), rightOperand.typeV2());
    }
    if (is(Kind.PLUS)
        && leftOperand.typeV2() instanceof ObjectType leftObjectType
        && leftObjectType.unwrappedType() instanceof ClassType leftClassType
        && rightOperand.typeV2() instanceof ObjectType rightObjectType
        && rightObjectType.unwrappedType() instanceof ClassType rightClassType
        && leftClassType == rightClassType) {
      return new ObjectType(leftClassType, TypeSource.min(leftObjectType.typeSource(), rightObjectType.typeSource()));
    }
    return PythonType.UNKNOWN;
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
