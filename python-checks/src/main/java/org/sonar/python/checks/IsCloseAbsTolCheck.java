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

import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6727")
public class IsCloseAbsTolCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Provide the \"abs_tol\" parameter when using \"math.isclose\" to compare a value to 0.";
  private static final String SECONDARY_LOCATION_MESSAGE = "This argument evaluates to zero.";
  private static final String QUICK_FIX_MESSAGE = "Add the \"abs_tol\" parameter.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR,
      ctx -> checkForIsCloseAbsTolArgument(ctx, (CallExpression) ctx.syntaxNode()));
  }

  private static void checkForIsCloseAbsTolArgument(SubscriptionContext ctx, CallExpression call) {
    Symbol symbol = call.calleeSymbol();
    if (symbol != null && "math.isclose".equals(symbol.fullyQualifiedName())
      && TreeUtils.argumentByKeyword("abs_tol", call.arguments()) == null) {
      RegularArgument firstArg = TreeUtils.nthArgumentOrKeyword(0, "a", call.arguments());
      RegularArgument secondArg = TreeUtils.nthArgumentOrKeyword(1, "b", call.arguments());
      checkArgumentExistsAndIsZero(firstArg)
        .ifPresentOrElse(
          argA -> addIssueAndQuickFix(argA, ctx, call),
          () -> checkArgumentExistsAndIsZero(secondArg).ifPresent(argB -> addIssueAndQuickFix(argB, ctx, call)));
    }
  }

  private static Optional<RegularArgument> checkArgumentExistsAndIsZero(@Nullable RegularArgument argument) {
    return Optional
      .ofNullable(argument)
      .filter(arg -> isLiteralZeroOrAssignedZero(arg.expression()));
  }

  private static void addIssueAndQuickFix(RegularArgument arg, SubscriptionContext ctx, CallExpression call) {
    PreciseIssue issue = ctx.addIssue(call.callee(), MESSAGE);
    issue.secondary(arg, SECONDARY_LOCATION_MESSAGE);

    PythonQuickFix quickFix = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
      .addTextEdit(TextEditUtils.insertBefore(call.rightPar(), ", abs_tol=1e-9"))
      .build();
    issue.addQuickFix(quickFix);
  }

  private static boolean isLiteralZeroOrAssignedZero(Expression expression) {
    return isZero(expression) || isAssignedZero(expression);
  }

  private static boolean isAssignedZero(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      Expression assignedValue = Expressions.singleAssignedValue((Name) expression);
      return assignedValue != null && isZero(assignedValue);
    }
    return false;
  }

  private static boolean isZero(Expression expression) {
    return expression.is(Tree.Kind.NUMERIC_LITERAL) && "0".equals(((NumericLiteral) (expression)).valueAsString());
  }
}
