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
package org.sonar.python.checks.hotspots;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.tree.TreeUtils;


class CommonValidationUtilsTest {

  class isLessThanTestCheck extends PythonSubscriptionCheck {
    @Override
    public void initialize(Context context) {
      context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, isLessThanTestCheck::checkCallExpr);
    }

    private static void checkCallExpr(SubscriptionContext ctx) {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      TreeUtils.nthArgumentOrKeywordOptional(0, "arg", callExpression.arguments())
        .ifPresent(argument -> {
          if (CommonValidationUtils.isLessThan(argument.expression(), 10)) {
            ctx.addIssue(argument, "Argument is less than 10");
          }
        });
    }
  }

  @Test
  void isLessThan() {
    PythonCheckVerifier.verify("src/test/resources/checks/commonValidationUtils.py", new isLessThanTestCheck());


  }
}