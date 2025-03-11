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
package org.sonar.python.semantic.v2.types;

import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.python.semantic.v2.TypeTable;
import org.sonar.python.tree.BinaryExpressionImpl;
import org.sonar.python.tree.UnaryExpressionImpl;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.plugins.python.api.types.v2.TypeSource;
import org.sonar.python.types.v2.TypeUtils;
import org.sonar.plugins.python.api.types.v2.UnionType;

public class TrivialTypePropagationVisitor extends BaseTreeVisitor {
  private final TypeCheckBuilder isBooleanTypeCheck;
  private final TypeCheckBuilder isIntTypeCheck;
  private final TypeCheckBuilder isFloatTypeCheck;
  private final TypeCheckBuilder isComplexTypeCheck;

  private final PythonType intType;
  private final PythonType boolType;

  public TrivialTypePropagationVisitor(TypeTable typeTable) {
    this.isBooleanTypeCheck = new TypeCheckBuilder(typeTable).isBuiltinWithName(BuiltinTypes.BOOL);
    this.isIntTypeCheck = new TypeCheckBuilder(typeTable).isBuiltinWithName(BuiltinTypes.INT);
    this.isFloatTypeCheck = new TypeCheckBuilder(typeTable).isBuiltinWithName(BuiltinTypes.FLOAT);
    this.isComplexTypeCheck = new TypeCheckBuilder(typeTable).isBuiltinWithName(BuiltinTypes.COMPLEX);

    var builtins = typeTable.getBuiltinsModule();
    this.intType = builtins.resolveMember(BuiltinTypes.INT).orElse(PythonType.UNKNOWN);
    this.boolType = builtins.resolveMember(BuiltinTypes.BOOL).orElse(PythonType.UNKNOWN);
  }

  @Override
  public void visitUnaryExpression(UnaryExpression unaryExpr) {
    super.visitUnaryExpression(unaryExpr);

    PythonType exprType = calculateUnaryExprType(unaryExpr);
    if (unaryExpr instanceof UnaryExpressionImpl unaryExprImpl) {
      unaryExprImpl.typeV2(toObjectType(exprType));
    }
  }

  @Override
  public void visitBinaryExpression(BinaryExpression binaryExpression) {
    super.visitBinaryExpression(binaryExpression);
    if (binaryExpression instanceof BinaryExpressionImpl binaryExpressionImpl) {
      var type = calculateBinaryExpressionType(binaryExpression);
      binaryExpressionImpl.typeV2(type);
    }
  }

  private static PythonType calculateBinaryExpressionType(BinaryExpression binaryExpression) {
    var kind = binaryExpression.getKind();
    var leftOperand = binaryExpression.leftOperand();
    var rightOperand = binaryExpression.rightOperand();
    if (binaryExpression.is(Tree.Kind.AND, Tree.Kind.OR)) {
      return UnionType.or(leftOperand.typeV2(), rightOperand.typeV2());
    }
    if (TypeDependenciesCalculator.SAME_TYPE_PRODUCING_BINARY_EXPRESSION_KINDS.contains(kind)
        && leftOperand.typeV2() instanceof ObjectType leftObjectType
        && leftObjectType.unwrappedType() instanceof ClassType leftClassType
        && rightOperand.typeV2() instanceof ObjectType rightObjectType
        && rightObjectType.unwrappedType() instanceof ClassType rightClassType
        && leftClassType == rightClassType) {
      return new ObjectType(leftClassType, TypeSource.min(leftObjectType.typeSource(), rightObjectType.typeSource()));
    }
    return PythonType.UNKNOWN;
  }

  private PythonType calculateUnaryExprType(UnaryExpression unaryExpr) {
    String operator = unaryExpr.operator().value();
    return TypeUtils.map(unaryExpr.expression().typeV2(), type -> mapUnaryExprType(operator, type));
  }

  private PythonType mapUnaryExprType(String operator, PythonType type) {
    return switch (operator) {
      case "~" -> mapInvertExprType(type);
      // not cannot be overloaded and always returns a boolean
      case "not" -> boolType;
      case "+", "-" -> mapUnaryPlusMinusType(type);
      default -> PythonType.UNKNOWN;
    };
  }

  private PythonType mapInvertExprType(PythonType type) {
    if(isIntTypeCheck.check(type) == TriBool.TRUE || isBooleanTypeCheck.check(type) == TriBool.TRUE) {
      return intType;
    }
    return PythonType.UNKNOWN;
  }

  private PythonType mapUnaryPlusMinusType(PythonType type) {
    if (isNumber(type)) {
      return type;
    } else if (isBooleanTypeCheck.check(type) == TriBool.TRUE) {
      return toObjectType(intType);
    }
    return PythonType.UNKNOWN;
  }

  private boolean isNumber(PythonType type) {
    return isIntTypeCheck.check(type) == TriBool.TRUE
      || isFloatTypeCheck.check(type) == TriBool.TRUE
      || isComplexTypeCheck.check(type) == TriBool.TRUE;
  }

  private static PythonType toObjectType(PythonType type) {
    if (type == PythonType.UNKNOWN) {
      return type;
    } else if (type instanceof ObjectType objectType) {
      return new ObjectType(objectType.typeWrapper(), objectType.attributes(), objectType.members(), objectType.typeSource());
    }
    return new ObjectType(type);
  }
}
