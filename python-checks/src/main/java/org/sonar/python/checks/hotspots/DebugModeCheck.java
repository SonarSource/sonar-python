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
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.PyArgumentTree;
import org.sonar.python.api.tree.PyAssignmentStatementTree;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyExpressionListTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyQualifiedExpressionTree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.semantic.Symbol;

@Rule(key = DebugModeCheck.CHECK_KEY)
public class DebugModeCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S4507";
  private static final String MESSAGE = "Make sure this debug feature is deactivated before delivering the code in production.";
  private static final Set<String> debugProperties = new HashSet<>(Arrays.asList("DEBUG", "DEBUG_PROPAGATE_EXCEPTIONS"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, ctx -> {
      PyCallExpressionTree callExpression = (PyCallExpressionTree) ctx.syntaxNode();
      List<PyArgumentTree> arguments = callExpression.arguments();
      if (!(callExpression.callee() instanceof PyQualifiedExpressionTree)) {
        return;
      }
      PyQualifiedExpressionTree callee = (PyQualifiedExpressionTree) callExpression.callee();
      if (getQualifiedName(callee.qualifier(), ctx).equals("django.conf.settings") && callee.name().name().equals("configure") && !arguments.isEmpty()) {
        arguments.stream().filter(DebugModeCheck::isDebugArgument).forEach(arg -> ctx.addIssue(arg, MESSAGE));
      }
    });

    context.registerSyntaxNodeConsumer(Kind.ASSIGNMENT_STMT, ctx -> {
      if (!ctx.pythonFile().fileName().equals("global_settings.py")) {
        return;
      }
      PyAssignmentStatementTree assignmentStatementTree = (PyAssignmentStatementTree) ctx.syntaxNode();
      for (PyExpressionListTree lhsExpression : assignmentStatementTree.lhsExpressions()) {
        boolean isDebugProperties = lhsExpression.expressions().stream().anyMatch(DebugModeCheck::isDebugIdentifier);
        if (isDebugProperties && assignmentStatementTree.assignedValues().stream().anyMatch(DebugModeCheck::isTrueLiteral)) {
          ctx.addIssue(assignmentStatementTree, MESSAGE);
        }
      }
    });
  }

  private static boolean isDebugIdentifier(PyExpressionTree expr) {
    return expr.is(Kind.NAME) && debugProperties.contains(((PyNameTree) expr).name());
  }

  private static boolean isTrueLiteral(PyExpressionTree expr) {
    return expr.is(Kind.NAME) && ((PyNameTree) expr).name().equals("True");
  }

  private static boolean isDebugArgument(PyArgumentTree argument) {
    PyNameTree keywordArgument = argument.keywordArgument();
    if (keywordArgument != null && debugProperties.contains((keywordArgument).name())) {
      return isTrueLiteral(argument.expression());
    }
    return false;
  }

  private static String getQualifiedName(PyExpressionTree node, SubscriptionContext ctx) {
    Symbol symbol = ctx.symbolTable().getSymbol(node);
    return symbol != null ? symbol.qualifiedName() : "";
  }
}
