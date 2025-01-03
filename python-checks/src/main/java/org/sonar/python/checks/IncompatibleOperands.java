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
package org.sonar.python.checks;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol.Parameter;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.types.InferredType;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.AMBIGUOUS;
import static org.sonar.plugins.python.api.symbols.Symbol.Kind.FUNCTION;
import static org.sonar.python.types.InferredTypes.typeSymbols;

public abstract class IncompatibleOperands extends PythonSubscriptionCheck {

  private static final Map<String, String> UNARY_SPECIAL_METHODS_BY_OPERATOR = new HashMap<>();
  static {
    UNARY_SPECIAL_METHODS_BY_OPERATOR.put("+", "__pos__");
    UNARY_SPECIAL_METHODS_BY_OPERATOR.put("-", "__neg__");
    UNARY_SPECIAL_METHODS_BY_OPERATOR.put("~", "__invert__");
  }

  private static final Map<String, SpecialMethodNames> BINARY_SPECIAL_METHODS_BY_OPERATOR = new HashMap<>();
  static {
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("<", new SpecialMethodNames("__lt__", "__gt__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("<=", new SpecialMethodNames("__le__", "__ge__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put(">", new SpecialMethodNames("__gt__", "__lt__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put(">=", new SpecialMethodNames("__ge__", "__le__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("+", new SpecialMethodNames("__add__", "__radd__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("-", new SpecialMethodNames("__sub__", "__rsub__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("*", new SpecialMethodNames("__mul__", "__rmul__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("@", new SpecialMethodNames("__matmul__", "__rmatmul__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("/", new SpecialMethodNames("__truediv__", "__rtruediv__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("//", new SpecialMethodNames("__floordiv__", "__rfloordiv__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("%", new SpecialMethodNames("__mod__", "__rmod__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("**", new SpecialMethodNames("__pow__", "__rpow__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("<<", new SpecialMethodNames("__lshift__", "__rlshift__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put(">>", new SpecialMethodNames("__rshift__", "__rrshift__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("&", new SpecialMethodNames("__and__", "__rand__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("^", new SpecialMethodNames("__xor__", "__rxor__"));
    BINARY_SPECIAL_METHODS_BY_OPERATOR.put("|", new SpecialMethodNames("__or__", "__ror__"));
  }

  private static class SpecialMethodNames {
    private final String left;
    private final String right;

    public SpecialMethodNames(String left, String right) {
      this.left = left;
      this.right = right;
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> ctx.syntaxNode().accept(new IncompatibleOperandsVisitor(ctx)));
  }

  private class IncompatibleOperandsVisitor extends BaseTreeVisitor {

    private final SubscriptionContext context;

    IncompatibleOperandsVisitor(SubscriptionContext context) {
      this.context = context;
    }

    @Override
    public void visitTypeAnnotation(TypeAnnotation tree) {
      // avoid raising FPs on type annotations
    }

    @Override
    public void visitBinaryExpression(BinaryExpression binaryExpression) {
      InferredType leftType = binaryExpression.leftOperand().type();
      InferredType rightType = binaryExpression.rightOperand().type();
      if (leftType.canOnlyBe("type") || rightType.canOnlyBe("type")) {
        // SONARPY-1666 We should only exclude types that represent ctypes
        return;
      }
      checkOperands(binaryExpression.operator(), leftType, rightType);
      super.visitBinaryExpression(binaryExpression);
    }

    @Override
    public void visitUnaryExpression(UnaryExpression unaryExpression) {
      InferredType type = unaryExpression.expression().type();
      Token operator = unaryExpression.operator();
      String memberName = UNARY_SPECIAL_METHODS_BY_OPERATOR.get(operator.value());
      if (memberName != null && resolveMethod(type, memberName).isAbsent()) {
        context.addIssue(operator, message(operator));
      }
      super.visitUnaryExpression(unaryExpression);
    }

    private void checkOperands(Token operator, InferredType left, InferredType right) {
      if (typeSymbols(left).stream().anyMatch(ClassSymbol::hasDecorators) || typeSymbols(right).stream().anyMatch(ClassSymbol::hasDecorators)) {
        return;
      }
      SpecialMethodNames specialMethodNames = BINARY_SPECIAL_METHODS_BY_OPERATOR.get(operator.value());
      if (specialMethodNames == null) {
        return;
      }
      SpecialMethod leftSpecialMethod = resolveMethod(left, specialMethodNames.left);
      SpecialMethod rightSpecialMethod = resolveMethod(right, specialMethodNames.right);
      if (leftSpecialMethod.isAbsent() && rightSpecialMethod.isAbsent()) {
        context.addIssue(operator, message(operator, left, right));
        return;
      }

      if (leftSpecialMethod.isUnresolved || rightSpecialMethod.isUnresolved) {
        return;
      }

      boolean hasIncompatibleLeftSpecialMethod = hasIncompatibleTypeOrAbsentSpecialMethod(leftSpecialMethod.symbol, right);
      boolean hasIncompatibleRightSpecialMethod = hasIncompatibleTypeOrAbsentSpecialMethod(rightSpecialMethod.symbol, left);
      if (hasIncompatibleLeftSpecialMethod && hasIncompatibleRightSpecialMethod) {
        context.addIssue(operator, message(operator, left, right));
      }
    }
  }

  private static boolean hasIncompatibleTypeOrAbsentSpecialMethod(@Nullable Symbol resolvedMethod, InferredType type) {
    if (resolvedMethod == null) {
      return true;
    }
    if (resolvedMethod.is(FUNCTION)) {
      List<Parameter> parameters = ((FunctionSymbol) resolvedMethod).parameters();
      if (parameters.size() == 2) {
        InferredType parameterType = parameters.get(1).declaredType();
        return !type.isCompatibleWith(parameterType);
      }
    }
    if (resolvedMethod.is(AMBIGUOUS)) {
      return ((AmbiguousSymbol) resolvedMethod).alternatives().stream().allMatch(s -> hasIncompatibleTypeOrAbsentSpecialMethod(s, type));
    }
    return false;
  }

  public abstract SpecialMethod resolveMethod(InferredType type, String method);
  public abstract String message(Token operator, InferredType left, InferredType right);
  public abstract String message(Token operator);

  protected static class SpecialMethod {
    @Nullable
    protected final Symbol symbol;
    private final boolean isUnresolved;

    public SpecialMethod(@Nullable Symbol symbol, boolean isUnresolved) {
      this.symbol = symbol;
      this.isUnresolved = isUnresolved;
    }

    boolean isAbsent() {
      return symbol == null && !isUnresolved;
    }
  }
}
