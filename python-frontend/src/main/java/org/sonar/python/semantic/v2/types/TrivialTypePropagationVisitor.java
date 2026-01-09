/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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

import java.util.EnumSet;
import java.util.Optional;
import java.util.Set;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.AwaitExpression;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.SliceExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeSource;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.python.semantic.v2.typetable.TypeTable;
import org.sonar.python.tree.AwaitExpressionImpl;
import org.sonar.python.tree.BinaryExpressionImpl;
import org.sonar.python.tree.CallExpressionImpl;
import org.sonar.python.tree.ConditionalExpressionImpl;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.tree.SliceExpressionImpl;
import org.sonar.python.tree.UnaryExpressionImpl;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeUtils;
import org.sonar.python.types.v2.matchers.TypePredicateContext;

public class TrivialTypePropagationVisitor extends BaseTreeVisitor {
  private static final Set<Tree.Kind> SAME_TYPE_PRODUCING_BINARY_EXPRESSION_KINDS = EnumSet.of(
    Tree.Kind.PLUS,
    Tree.Kind.MINUS,
    Tree.Kind.MULTIPLICATION,
    Tree.Kind.DIVISION,
    Tree.Kind.FLOOR_DIVISION,
    Tree.Kind.MODULO,
    Tree.Kind.POWER);

  private static final TypeInferenceMatcher IS_SLICEABLE_TYPE = TypeInferenceMatcher.of(
    TypeInferenceMatchers.any(
      TypeInferenceMatchers.isObjectOfType(BuiltinTypes.LIST),
      TypeInferenceMatchers.isObjectOfType(BuiltinTypes.TUPLE),
      TypeInferenceMatchers.isObjectOfType(BuiltinTypes.STR)));

  private final TypeCheckBuilder isBooleanTypeCheck;
  private final TypeCheckBuilder isIntTypeCheck;
  private final TypeCheckBuilder isFloatTypeCheck;
  private final TypeCheckBuilder isComplexTypeCheck;
  private final TypeCheckBuilder isPropertyTypeCheck;

  private final PythonType intType;
  private final PythonType boolType;

  private final TypePredicateContext typePredicateContext;
  private final AwaitedTypeCalculator awaitedTypeCalculator;

  public TrivialTypePropagationVisitor(TypeTable typeTable) {
    this.isBooleanTypeCheck = new TypeCheckBuilder(typeTable).isBuiltinWithName(BuiltinTypes.BOOL);
    this.isIntTypeCheck = new TypeCheckBuilder(typeTable).isBuiltinWithName(BuiltinTypes.INT);
    this.isFloatTypeCheck = new TypeCheckBuilder(typeTable).isBuiltinWithName(BuiltinTypes.FLOAT);
    this.isComplexTypeCheck = new TypeCheckBuilder(typeTable).isBuiltinWithName(BuiltinTypes.COMPLEX);
    this.isPropertyTypeCheck = new TypeCheckBuilder(typeTable).isSubtypeOf("property");

    var builtins = typeTable.getBuiltinsModule();
    this.intType = builtins.resolveMember(BuiltinTypes.INT).orElse(PythonType.UNKNOWN);
    this.boolType = builtins.resolveMember(BuiltinTypes.BOOL).orElse(PythonType.UNKNOWN);

    this.typePredicateContext = TypePredicateContext.of(typeTable);
    this.awaitedTypeCalculator = new AwaitedTypeCalculator(typeTable);
  }

  @Override
  public void visitQualifiedExpression(QualifiedExpression qualifiedExpression) {
    scan(qualifiedExpression.qualifier());
    if (qualifiedExpression.name() instanceof NameImpl name) {
      Optional<PythonType> pythonType = Optional.of(qualifiedExpression.qualifier())
        .map(Expression::typeV2)
        .flatMap(t -> t.resolveMember(name.name()));
      if (pythonType.isPresent()) {
        var type = pythonType.get();
        if (type instanceof FunctionType functionType) {
          // If a member access is a method with a "property" annotation, we consider the resulting type to be the return type of the method
          boolean isProperty = functionType.decorators().stream().anyMatch(t -> isPropertyTypeCheck.check(t.type()) == TriBool.TRUE);
          if (isProperty) {
            type = functionType.returnType();
          }
        }
        name.typeV2(type);
      } else {
        name.typeV2(PythonType.UNKNOWN);
      }
    }
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

  @Override
  public void visitCallExpression(CallExpression callExpr) {
    super.visitCallExpression(callExpr);
    if (callExpr instanceof CallExpressionImpl callExprImpl) {
      PythonType type = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);
      callExprImpl.typeV2(type);
    }
  }

  @Override
  public void visitAwaitExpression(AwaitExpression awaitExpression) {
    super.visitAwaitExpression(awaitExpression);
    PythonType awaitedType = awaitExpression.expression().typeV2();
    PythonType resultType = awaitedTypeCalculator.calculate(awaitedType);
    if (awaitExpression instanceof AwaitExpressionImpl awaitExpressionImpl) {
      awaitExpressionImpl.typeV2(resultType);
    }
  }

  @Override
  public void visitConditionalExpression(ConditionalExpression conditionalExpression) {
    super.visitConditionalExpression(conditionalExpression);
    if (conditionalExpression instanceof ConditionalExpressionImpl conditionalExpressionImpl) {
      PythonType trueType = conditionalExpression.trueExpression().typeV2();
      PythonType falseType = conditionalExpression.falseExpression().typeV2();
      conditionalExpressionImpl.typeV2(UnionType.or(trueType, falseType));
    }
  }

  @Override
  public void visitSliceExpression(SliceExpression sliceExpression) {
    super.visitSliceExpression(sliceExpression);
    if (sliceExpression instanceof SliceExpressionImpl sliceExpressionImpl) {
      PythonType objectType = sliceExpression.object().typeV2();
      sliceExpressionImpl.typeV2(calculateSliceExpressionType(objectType));
    }
  }

  private PythonType calculateSliceExpressionType(PythonType objectType) {
    if (IS_SLICEABLE_TYPE.evaluate(objectType, typePredicateContext) == TriBool.TRUE) {
      return objectType;
    }
    return PythonType.UNKNOWN;
  }

  private static PythonType calculateBinaryExpressionType(BinaryExpression binaryExpression) {
    var kind = binaryExpression.getKind();
    var leftOperand = binaryExpression.leftOperand();
    var rightOperand = binaryExpression.rightOperand();
    if (binaryExpression.is(Tree.Kind.AND, Tree.Kind.OR)) {
      return UnionType.or(leftOperand.typeV2(), rightOperand.typeV2());
    }
    // preserve union types set by TrivialTypeInferenceVisitor
    if (binaryExpression.is(Tree.Kind.BITWISE_OR) && binaryExpression.typeV2() instanceof UnionType) {
      return binaryExpression.typeV2();
    }
    if (TrivialTypePropagationVisitor.SAME_TYPE_PRODUCING_BINARY_EXPRESSION_KINDS.contains(kind)
        && leftOperand.typeV2() instanceof ObjectType leftObjectType
        && leftObjectType.unwrappedType() instanceof ClassType leftClassType
        && rightOperand.typeV2() instanceof ObjectType rightObjectType
        && rightObjectType.unwrappedType() instanceof ClassType rightClassType
        && leftClassType == rightClassType) {
      return ObjectType.Builder.fromType(leftClassType)
        .withTypeSource(TypeSource.min(leftObjectType.typeSource(), rightObjectType.typeSource()))
        .build();
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
      return ObjectType.Builder.fromType(objectType.type())
        .withAttributes(objectType.attributes())
        .withMembers(objectType.members())
        .withTypeSource(objectType.typeSource())
        .build();
    }
    return ObjectType.fromType(type);
  }

}
