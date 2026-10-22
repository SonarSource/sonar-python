/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.TypeShed;

@Rule(key = "S5607")
public class IncompatibleOperandsCheck extends IncompatibleOperands {

  private static final TypeMatcher OBJECT = TypeMatchers.isType("builtins.object");
  private static final TypeMatcher OBJECT_DIR = TypeMatchers.isType("builtins.object.__dir__");

  @Override
  protected InferredType operandType(Expression operand, SubscriptionContext context) {
    Expression expression = Expressions.removeParentheses(operand);
    if (expression instanceof CallExpression call && isBuiltinDirCall(call, context)) {
      // The builtin implementation returns a list. Its stub allows Iterable[str]
      // so subclasses can override __dir__ with other iterable return types.
      return InferredTypes.runtimeType(TypeShed.typeShedClass("list"));
    }
    return super.operandType(operand, context);
  }

  private static boolean isBuiltinDirCall(CallExpression call, SubscriptionContext context) {
    Expression callee = Expressions.removeParentheses(call.callee());
    if (callee instanceof Name name) {
      callee = Expressions.singleAssignedNonNameValue(name).orElse(callee);
    }
    // A call through an instance may dispatch to an overriding implementation.
    return callee instanceof QualifiedExpression member
      && OBJECT.isTrueFor(member.qualifier(), context)
      && OBJECT_DIR.isTrueFor(call.callee(), context);
  }

  @Override
  public SpecialMethod resolveMethod(InferredType type, String method) {
    Symbol resolvedMethod = type.resolveMember(method).orElse(null);
    boolean isUnresolved = resolvedMethod == null && type.canHaveMember(method);
    return new SpecialMethod(resolvedMethod, isUnresolved);
  }

  @Override
  public String message(Token operator, InferredType left, InferredType right) {
    String leftTypeName = InferredTypes.typeName(left);
    String rightTypeName = InferredTypes.typeName(right);
    String message = "Fix this invalid \"" + operator.value() + "\" operation between incompatible types";
    if (leftTypeName != null && rightTypeName != null) {
      message += " (" + leftTypeName + " and " + rightTypeName + ")";
    }
    return message + ".";
  }

  @Override
  public String message(Token operator) {
    return "Fix this invalid \"" + operator.value() + "\" operation on a type which doesn't support it.";
  }
}
