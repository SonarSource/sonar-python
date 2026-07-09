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
package org.sonar.python.checks.tests;

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.UnittestUtils;
import org.sonar.python.checks.utils.UnittestUtils.AssertionArguments;
import org.sonar.python.checks.utils.UnittestUtils.AssertionFrameworkHandlers;

import static org.sonar.python.checks.utils.UnittestUtils.ASSERTPY_EQUALITY_ASSERTION_MATCHER;

@Rule(key = "S5863")
public class IdenticalAssertionArgumentsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this assertion to not have the same actual and expected expression.";
  private static final String SECONDARY_ACTUAL_MESSAGE = "This is the same expression as the expected argument.";
  private static final String SECONDARY_ASSIGNMENT_MESSAGE = "Assigned here.";
  private static final Set<String> PYTEST_OPERATORS = Set.of("==", "!=");

  @Override
  public void initialize(Context context) {
    UnittestUtils.registerAssertionSyntaxNodeConsumers(context, new AssertionFrameworkHandlers(
      IdenticalAssertionArgumentsCheck::checkUnittestAssertion,
      IdenticalAssertionArgumentsCheck::checkAssertpyAssertion,
      IdenticalAssertionArgumentsCheck::checkPytestAssertion));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.TESTS;
  }

  private static void checkUnittestAssertion(SubscriptionContext ctx, CallExpression callExpression) {
    AssertionArguments arguments = UnittestUtils.unittestAssertionArguments(callExpression, ctx);
    if (arguments != null) {
      reportIfIdenticalArguments(ctx, arguments);
    }
  }

  private static void checkPytestAssertion(SubscriptionContext ctx, AssertStatement assertStatement) {
    if (!UnittestUtils.isPytestStyleTestFunction(ctx, assertStatement)) {
      return;
    }

    Expression condition = Expressions.removeParentheses(assertStatement.condition());
    if (!(condition instanceof BinaryExpression binaryExpression)
      || !PYTEST_OPERATORS.contains(binaryExpression.operator().value())) {
      return;
    }

    reportIfIdenticalArguments(ctx, new AssertionArguments(binaryExpression.leftOperand(), binaryExpression.rightOperand()));
  }

  private static void checkAssertpyAssertion(SubscriptionContext ctx, CallExpression callExpression) {
    AssertionArguments arguments = UnittestUtils.assertpyAssertionArguments(callExpression, ctx, ASSERTPY_EQUALITY_ASSERTION_MATCHER);
    if (arguments != null) {
      reportIfIdenticalArguments(ctx, arguments);
    }
  }

  private static void reportIfIdenticalArguments(SubscriptionContext ctx, AssertionArguments arguments) {
    if (CheckUtils.areEquivalent(arguments.actual(), arguments.expected())) {
      var issue = ctx.addIssue(arguments.expected(), MESSAGE)
        .secondary(arguments.actual(), SECONDARY_ACTUAL_MESSAGE);
      if (arguments.actual() instanceof Name name) {
        Expression assignment = Expressions.singleAssignedValue(name);
        if (assignment != null) {
          issue.secondary(assignment, SECONDARY_ASSIGNMENT_MESSAGE);
        }
      }
    }
  }
}
