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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S3358")
public class NestedConditionalExpressionCheck extends PythonSubscriptionCheck {

  private static final Kind[] COMPREHENSION_KINDS = {
    Kind.LIST_COMPREHENSION,
    Kind.DICT_COMPREHENSION,
    Kind.SET_COMPREHENSION,
    Kind.GENERATOR_EXPR
  };

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.CONDITIONAL_EXPR, ctx -> {
      Tree conditionalExpression = ctx.syntaxNode();
      Tree parentConditional = TreeUtils.firstAncestorOfKind(conditionalExpression, Kind.CONDITIONAL_EXPR);
      if (parentConditional != null) {
        boolean isInsideComprehension = TreeUtils.firstAncestorOfKind(conditionalExpression, COMPREHENSION_KINDS) != null;
        if (!isInsideComprehension) {
          ctx.addIssue(conditionalExpression, "Extract this nested conditional expression into an independent statement.")
            .secondary(((ConditionalExpression) parentConditional).ifKeyword(), "Parent conditional expression.");
        }
      }
    });
  }

}
