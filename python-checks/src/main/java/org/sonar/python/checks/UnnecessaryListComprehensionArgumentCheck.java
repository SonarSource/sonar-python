/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7492")
public class UnnecessaryListComprehensionArgumentCheck extends PythonSubscriptionCheck {
  private TypeCheckBuilder isAllTypeCheck;
  private TypeCheckBuilder isAnyTypeCheck;
  private TypeCheckBuilder isListTypeCheck;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::check);
  }

  private void initChecks(SubscriptionContext ctx) {
    isAllTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithName("all");
    isAnyTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithName("any");
    isListTypeCheck = ctx.typeChecker().typeCheckBuilder().isInstanceOf("list");
  }

  private void check(SubscriptionContext ctx) {
    var callExpression = (CallExpression) ctx.syntaxNode();
    if (!isSensitiveCall(callExpression)) {
      return;
    }
    TreeUtils.nthArgumentOrKeywordOptional(0, "", callExpression.arguments())
      .map(RegularArgument::expression)
      .ifPresent(argumentExpression -> {
        if (isListComprehensionExpression(argumentExpression) || isAssignedToListComprehensionExpression(argumentExpression)) {
          ctx.addIssue(callExpression, "Unpack this comprehension expression");
        }
      });
  }

  private boolean isListComprehensionExpression(@Nullable Expression expression) {
    return expression instanceof ComprehensionExpression comprehensionExpression
           && isListTypeCheck.check(comprehensionExpression.typeV2()) == TriBool.TRUE;
  }

  private boolean isAssignedToListComprehensionExpression(Expression argumentExpression) {
    return argumentExpression instanceof Name name
           && getUsageCount(name) == 2
           && isListComprehensionExpression(Expressions.singleAssignedValue(name));
  }

  private boolean isSensitiveCall(CallExpression callExpression) {
    return Stream.of(isAllTypeCheck, isAnyTypeCheck)
      .map(check -> check.check(callExpression.callee().typeV2()))
      .anyMatch(TriBool.TRUE::equals);
  }

  private static int getUsageCount(Name name) {
    return Optional.of(name)
      .map(Name::symbolV2)
      .map(SymbolV2::usages)
      .map(List::size)
      .orElse(0);
  }
}
