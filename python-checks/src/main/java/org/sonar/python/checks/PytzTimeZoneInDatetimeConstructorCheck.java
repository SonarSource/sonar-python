/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6887")
public class PytzTimeZoneInDatetimeConstructorCheck extends PythonSubscriptionCheck {

  private ReachingDefinitionsAnalysis reachingDefinitionsAnalysis;
  private static final String MESSAGE = "Don't pass a \"pytz.timezone\" to the \"datetime.datetime\" constructor.";
  private static final String SECONDARY_MESSAGE = "The pytz.timezone is created here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT,
      ctx -> reachingDefinitionsAnalysis = new ReachingDefinitionsAnalysis(ctx.pythonFile()));

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpression);
  }

  private void checkCallExpression(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();

    if (calleeSymbol != null && "datetime.datetime".equals(calleeSymbol.fullyQualifiedName())) {
      RegularArgument argument = TreeUtils.nthArgumentOrKeyword(7, "tzinfo", callExpression.arguments());
      if (argument == null) {
        return;
      }
      checkArgument(argument, context);
    }
  }

  private void checkArgument(RegularArgument argument, SubscriptionContext context) {
    if (argument.expression().is(Tree.Kind.CALL_EXPR)) {
      CallExpression callExpression = (CallExpression) argument.expression();
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (!(calleeSymbol != null && "pytz.timezone".equals(calleeSymbol.fullyQualifiedName()))) {
        return;
      }
      context.addIssue(argument, MESSAGE);
    } else if (argument.expression().is(Tree.Kind.NAME)) {
      List<CallExpression> allSecondaryLocations = reachingDefinitionsAnalysis.valuesAtLocation((Name) argument.expression()).stream()
        .filter(expression -> expression.is(Tree.Kind.CALL_EXPR))
        .map(CallExpression.class::cast)
        .filter(call -> Optional.ofNullable(call.calleeSymbol()).map(symbol ->"pytz.timezone".equals(symbol.fullyQualifiedName())).orElse(false))
        .sorted(Comparator.comparingInt(call -> call.firstToken().line()))
        .toList();

      if (allSecondaryLocations.isEmpty()) {
        return;
      }
      var issue = context.addIssue(argument, MESSAGE);
      allSecondaryLocations.forEach(secondaryLocation -> issue.secondary(secondaryLocation, SECONDARY_MESSAGE));
    }
  }
}
