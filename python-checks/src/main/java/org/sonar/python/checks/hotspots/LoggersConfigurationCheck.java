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

import java.util.Arrays;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.symbols.Symbol;

@Rule(key = "S4792")
public class LoggersConfigurationCheck extends PythonSubscriptionCheck {

  private static final List<String> FUNCTIONS_TO_CHECK = Arrays.asList(
    "logging.basicConfig",
    "logging.disable",
    "logging.setLoggerClass",
    "logging.config.fileConfig",
    "logging.config.dictConfig");

  private static final List<String> LOGGERS_CLASSES = Arrays.asList(
    "logging.Logger",
    "logging.Handler",
    "logging.Filter");

  private static final String MESSAGE = "Make sure that this logger's configuration is safe.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpressionTree = (CallExpression) ctx.syntaxNode();
      Symbol symbol = callExpressionTree.calleeSymbol();
      if (symbol != null && FUNCTIONS_TO_CHECK.contains(symbol.fullyQualifiedName())) {
        ctx.addIssue(callExpressionTree, MESSAGE);
      }
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, ctx -> isSettingLastResort(ctx, (AssignmentStatement) ctx.syntaxNode()));

    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> isClassExtendingLogger(ctx, (ClassDef) ctx.syntaxNode()));
  }

  private static void isClassExtendingLogger(SubscriptionContext ctx, ClassDef classDef) {
    ArgList argList = classDef.args();
    if (argList != null) {
      argList.arguments().stream()
        .filter(argument -> argument.is(Tree.Kind.REGULAR_ARGUMENT))
        .map(RegularArgument.class::cast)
        .map(RegularArgument::expression)
        .filter(expr -> expr instanceof HasSymbol hasSymbol && hasSymbol.symbol() != null)
        .filter(expr -> LOGGERS_CLASSES.contains(((HasSymbol) expr).symbol().fullyQualifiedName()))
        .forEach(expr -> ctx.addIssue(expr, MESSAGE));
    }
  }

  // check if logging.lastResort is being set
  private static void isSettingLastResort(SubscriptionContext ctx, AssignmentStatement assignment) {
    assignment.lhsExpressions().stream()
      .flatMap(exprList -> exprList.expressions().stream())
      .forEach(expr -> {
        if (expr instanceof HasSymbol hasSymbol) {
          Symbol symbol = hasSymbol.symbol();
          if (symbol != null && "logging.lastResort".equals(symbol.fullyQualifiedName())) {
            ctx.addIssue(expr, MESSAGE);
          }
        }
      });
  }
}
