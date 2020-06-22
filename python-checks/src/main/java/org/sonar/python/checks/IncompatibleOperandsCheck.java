/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.checks;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol.Parameter;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.types.InferredType;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.FUNCTION;
import static org.sonar.plugins.python.api.types.BuiltinTypes.COMPLEX;
import static org.sonar.plugins.python.api.types.BuiltinTypes.FLOAT;
import static org.sonar.plugins.python.api.types.BuiltinTypes.INT;
import static org.sonar.plugins.python.api.types.BuiltinTypes.STR;
import static org.sonar.python.types.InferredTypes.typeSymbols;

@Rule(key = "S5607")
public class IncompatibleOperandsCheck extends PythonVisitorCheck {

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

  // https://github.com/python/mypy/blob/e97377c454a1d5c019e9c56871d5f229db6b47b2/mypy/semanal_classprop.py#L16-L46
  private static final Set<Set<String>> HARDCODED_COMPATIBLE_TYPES = new HashSet<>();
  static {
    HARDCODED_COMPATIBLE_TYPES.add(new HashSet<>(Arrays.asList(INT, FLOAT, COMPLEX)));
    HARDCODED_COMPATIBLE_TYPES.add(new HashSet<>(Arrays.asList(STR, "unicode", "bytearray", "bytes", "memoryview")));
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
  public void visitBinaryExpression(BinaryExpression binaryExpression) {
    InferredType leftType = binaryExpression.leftOperand().type();
    InferredType rightType = binaryExpression.rightOperand().type();
    checkOperands(binaryExpression.operator(), leftType, rightType);
    super.visitBinaryExpression(binaryExpression);
  }

  @Override
  public void visitUnaryExpression(UnaryExpression unaryExpression) {
    InferredType type = unaryExpression.expression().type();
    Token operator = unaryExpression.operator();
    String memberName = UNARY_SPECIAL_METHODS_BY_OPERATOR.get(operator.value());
    if (memberName != null && !type.canHaveMember(memberName)) {
      addIssue(operator, "Fix this invalid \"" + operator.value() + "\" operation on a type which doesn't support it.");
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
    boolean canHaveLeftSpecialMethod = left.canHaveMember(specialMethodNames.left);
    boolean canHaveRightSpecialMethod = right.canHaveMember(specialMethodNames.right);
    if (!canHaveLeftSpecialMethod && !canHaveRightSpecialMethod) {
      addIssue(operator, message(operator));
      return;
    }

    Symbol leftSpecialMethod = left.resolveMember(specialMethodNames.left).orElse(null);
    Symbol rightSpecialMethod = right.resolveMember(specialMethodNames.right).orElse(null);
    if (hasUnresolvedMethod(canHaveLeftSpecialMethod, leftSpecialMethod) || hasUnresolvedMethod(canHaveRightSpecialMethod, rightSpecialMethod)) {
      return;
    }

    boolean hasIncompatibleLeftSpecialMethod = hasIncompatibleTypeOrAbsentSpecialMethod(leftSpecialMethod, right);
    boolean hasIncompatibleRightSpecialMethod = hasIncompatibleTypeOrAbsentSpecialMethod(rightSpecialMethod, left);
    if (hasIncompatibleLeftSpecialMethod && hasIncompatibleRightSpecialMethod) {
      addIssue(operator, message(operator));
    }
  }

  private static String message(Token operator) {
    return "Fix this invalid \"" + operator.value() + "\" operation between incompatible types.";
  }

  private static boolean hasUnresolvedMethod(boolean canHaveMethod, @Nullable Symbol method) {
    return canHaveMethod && method == null;
  }

  private static boolean hasIncompatibleTypeOrAbsentSpecialMethod(@Nullable Symbol resolvedMethod, InferredType type) {
    if (resolvedMethod == null) {
      return true;
    }
    if (resolvedMethod.is(FUNCTION)) {
      List<Parameter> parameters = ((FunctionSymbol) resolvedMethod).parameters();
      if (parameters.size() == 2) {
        InferredType parameterType = parameters.get(1).declaredType();
        return !areHardcodedCompatible(type, parameterType) && !type.isCompatibleWith(parameterType);
      }
    }
    return false;
  }

  private static boolean areHardcodedCompatible(InferredType type, InferredType parameterType) {
    return HARDCODED_COMPATIBLE_TYPES.stream().anyMatch(compatibleTypes ->
      compatibleTypes.stream().anyMatch(type::canBeOrExtend) && compatibleTypes.stream().anyMatch(parameterType::canBeOrExtend));
  }
}
