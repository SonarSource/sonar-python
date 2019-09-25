/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks.hotspots;

import java.util.Arrays;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.HasSymbol;
import org.sonar.python.api.tree.PyArgListTree;
import org.sonar.python.api.tree.PyArgumentTree;
import org.sonar.python.api.tree.PyAssignmentStatementTree;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyClassDefTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.semantic.TreeSymbol;

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
      PyCallExpressionTree callExpressionTree = (PyCallExpressionTree) ctx.syntaxNode();
      TreeSymbol symbol = callExpressionTree.calleeSymbol();
      if (symbol != null && FUNCTIONS_TO_CHECK.contains(symbol.fullyQualifiedName())) {
        ctx.addIssue(callExpressionTree, MESSAGE);
      }
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, ctx -> isSettingLastResort(ctx, (PyAssignmentStatementTree) ctx.syntaxNode()));

    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> isClassExtendingLogger(ctx, (PyClassDefTree) ctx.syntaxNode()));
  }

  private static void isClassExtendingLogger(SubscriptionContext ctx, PyClassDefTree classDef) {
    PyArgListTree argList = classDef.args();
    if (argList != null) {
      argList.arguments().stream()
        .map(PyArgumentTree::expression)
        .filter(expr -> expr instanceof HasSymbol && ((HasSymbol) expr).symbol() != null)
        .filter(expr -> LOGGERS_CLASSES.contains(((HasSymbol) expr).symbol().fullyQualifiedName()))
        .forEach(expr -> ctx.addIssue(expr, MESSAGE));
    }
  }

  // check if logging.lastResort is being set
  private static void isSettingLastResort(SubscriptionContext ctx, PyAssignmentStatementTree assignment) {
    assignment.lhsExpressions().stream()
      .flatMap(exprList -> exprList.expressions().stream())
      .forEach(expr -> {
        if (expr instanceof HasSymbol) {
          TreeSymbol symbol = ((HasSymbol) expr).symbol();
          if (symbol != null && "logging.lastResort".equals(symbol.fullyQualifiedName())) {
            ctx.addIssue(expr, MESSAGE);
          }
        }
      });
  }
}
