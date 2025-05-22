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

import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7510")
public class UnnecessaryReversedCallCheck extends PythonSubscriptionCheck {
  private TypeCheckBuilder isReversedTypeCheck;
  private TypeCheckBuilder isSortedTypeCheck;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::check);
  }

  private void initChecks(SubscriptionContext ctx) {
    isReversedTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithName("reversed");
    isSortedTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithName("sorted");
  }

  private void check(SubscriptionContext ctx) {
    if (ctx.syntaxNode() instanceof CallExpression callExpression && isReversedCall(callExpression)) {
      TreeUtils.nthArgumentOrKeywordOptional(0, "", callExpression.arguments())
        .map(RegularArgument::expression)
        .ifPresent(argumentExpression -> {
          if (isSortedCall(argumentExpression) || isAssignedToSortedCall(argumentExpression)) {
            ctx.addIssue(callExpression, "Remove this redundant reversed call, use reverse argument of the sorted function call instead");
          }
        });
    }
  }

  private boolean isAssignedToSortedCall(Expression argumentExpression) {
    return argumentExpression instanceof Name name
           && getUsageCount(name) == 2
           && isSortedCall(Expressions.singleAssignedValue(name));
  }

  private boolean isReversedCall(CallExpression callExpression) {
    return isReversedTypeCheck.check(callExpression.callee().typeV2()) == TriBool.TRUE;
  }

  private boolean isSortedCall(@Nullable Expression expression) {
    return expression instanceof CallExpression callExpression && isSortedTypeCheck.check(callExpression.callee().typeV2()) == TriBool.TRUE;
  }

  private static int getUsageCount(Name name) {
    var symbol = name.symbolV2();
    if (symbol == null) return 0;
    return symbol.usages().size();
  }
}
