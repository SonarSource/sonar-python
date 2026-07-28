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

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.UnittestUtils;

@Rule(key = "S9078")
public class DuplicateParametrizeCasesCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove this duplicate test case.";
  private static final String SECONDARY_MESSAGE = "Original.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DECORATOR, DuplicateParametrizeCasesCheck::checkDecorator);
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkDecorator(SubscriptionContext ctx) {
    Decorator decorator = (Decorator) ctx.syntaxNode();
    Expression valuesExpression = UnittestUtils.parametrizeArgvaluesExpression(decorator, ctx);
    if (valuesExpression == null || !valuesExpression.is(Tree.Kind.LIST_LITERAL, Tree.Kind.TUPLE)) {
      return;
    }

    raiseOnDuplicates(Expressions.expressionsFromListOrTuple(valuesExpression), ctx);
  }

  private static void raiseOnDuplicates(List<Expression> cases, SubscriptionContext ctx) {
    List<Expression> distinctCases = new ArrayList<>();
    for (Expression testCase : cases) {
      Expression original = findEquivalent(distinctCases, testCase);
      if (original != null) {
        ctx.addIssue(testCase, MESSAGE).secondary(original, SECONDARY_MESSAGE);
      } else {
        distinctCases.add(testCase);
      }
    }
  }

  private static Expression findEquivalent(List<Expression> distinctCases, Expression testCase) {
    Expression normalizedCase = Expressions.removeParentheses(testCase);
    for (Expression distinctCase : distinctCases) {
      if (CheckUtils.areEquivalent(Expressions.removeParentheses(distinctCase), normalizedCase)) {
        return distinctCase;
      }
    }
    return null;
  }
}
