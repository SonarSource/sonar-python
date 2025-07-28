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

import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7505")
public class UnnecessaryLambdaMapCallCheck extends PythonSubscriptionCheck {
  private TypeCheckBuilder isMapTypeCheck;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::check);
  }

  private void initChecks(SubscriptionContext ctx) {
    isMapTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithName("map");
  }

  private void check(SubscriptionContext ctx) {
    var callExpression = (CallExpression) ctx.syntaxNode();
    if (isMapCall(callExpression)) {
      TreeUtils.nthArgumentOrKeywordOptional(0, "", callExpression.arguments())
        .map(RegularArgument::expression)
        .ifPresent(argumentExpression -> {
          if (isLambda(argumentExpression) || isAssignedToLambda(argumentExpression)) {
            ctx.addIssue(callExpression, "Replace this map call with a comprehension.");
          }
        });
    }
  }

  private boolean isMapCall(CallExpression callExpression) {
    return isMapTypeCheck.check(callExpression.callee().typeV2()) == TriBool.TRUE;
  }

  private static boolean isLambda(@Nullable Expression argumentExpression) {
    return argumentExpression instanceof LambdaExpression;
  }

  private static boolean isAssignedToLambda(Expression argumentExpression) {
    return argumentExpression instanceof Name name
           && getUsageCount(name) == 2
           && isLambda(Expressions.singleAssignedValue(name));
  }

  private static int getUsageCount(Name name) {
    return Optional.ofNullable(name.symbolV2())
      .map(SymbolV2::usages)
      .map(List::size)
      .orElse(0);
  }
}
