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
package org.sonar.python.checks.tests;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S1607")
public class SkippedTestNoReasonCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Provide a reason for skipping this test.";

  private static final Set<String> skipDecoratorsFQN = new HashSet<>(List.of("unittest.case.skip", "pytest.mark.skip"));
  private static final Set<String> skipCallExpressionsFQN = new HashSet<>(List.of("pytest.skip"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DECORATOR, ctx -> {
      Decorator decorator = (Decorator) ctx.syntaxNode();
      checkDecoratorSkipWithoutReason(ctx, decorator);
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      checkCallExpressionSkipWithNoOrEmptyReason(ctx, callExpression);
    });
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkDecoratorSkipWithoutReason(SubscriptionContext ctx, Decorator decorator) {
    Expression expression = decorator.expression();
    Symbol symbol = getSymbolFromExpression(expression);

    if (symbol == null) {
      return;
    }

    if (!skipDecoratorsFQN.contains(symbol.fullyQualifiedName())) {
      return;
    }

    checkNoOrEmptyReason(ctx, decorator, decorator.arguments());
  }

  private static Symbol getSymbolFromExpression(Expression expression) {
    if (expression instanceof HasSymbol hasSymbol) {
      return hasSymbol.symbol();
    }

    if (expression.is(Tree.Kind.CALL_EXPR)) {
      return ((CallExpression) expression).calleeSymbol();
    }

    return null;
  }

  private static void checkCallExpressionSkipWithNoOrEmptyReason(SubscriptionContext ctx, CallExpression callExpression) {
    Symbol symbol = (callExpression.calleeSymbol());
    if (symbol == null) {
      return;
    }

    if (!skipCallExpressionsFQN.contains(symbol.fullyQualifiedName())) {
      return;
    }

    checkNoOrEmptyReason(ctx, callExpression, callExpression.argumentList());
  }

  private static void checkNoOrEmptyReason(SubscriptionContext ctx, Tree node, ArgList args) {
    if (args == null) {
      ctx.addIssue(node, MESSAGE);
      return;
    }

    Argument arg = args.arguments().get(0);
    if (!arg.is(Tree.Kind.REGULAR_ARGUMENT)) {
      return;
    }

    RegularArgument regularArg = (RegularArgument) arg;
    if (!regularArg.expression().is(Tree.Kind.STRING_LITERAL)) {
      return;
    }

    StringLiteral stringLiteral = (StringLiteral) regularArg.expression();

    if (stringLiteral.trimmedQuotesValue().equals("")) {
      ctx.addIssue(stringLiteral, MESSAGE);
    }
  }
}
