/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.symbols.Symbol;

@Rule(key = DebugModeCheck.CHECK_KEY)
public class DebugModeCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S4507";
  private static final String MESSAGE = "Make sure this debug feature is deactivated before delivering the code in production.";
  private static final List<String> debugProperties = Arrays.asList("DEBUG", "DEBUG_PROPAGATE_EXCEPTIONS");
  private static final List<String> settingFiles = Arrays.asList("global_settings.py", "settings.py");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      List<Argument> arguments = callExpression.arguments();
      if (!(callExpression.callee() instanceof QualifiedExpression)) {
        return;
      }
      if ("django.conf.settings.configure".equals(getQualifiedName(callExpression)) && !arguments.isEmpty()) {
        arguments.stream().filter(DebugModeCheck::isDebugArgument).forEach(arg -> ctx.addIssue(arg, MESSAGE));
      }
    });

    context.registerSyntaxNodeConsumer(Kind.ASSIGNMENT_STMT, ctx -> {
      if (!settingFiles.contains(ctx.pythonFile().fileName())) {
        return;
      }
      AssignmentStatement assignmentStatementTree = (AssignmentStatement) ctx.syntaxNode();
      for (ExpressionList lhsExpression : assignmentStatementTree.lhsExpressions()) {
        boolean isDebugProperties = lhsExpression.expressions().stream().anyMatch(DebugModeCheck::isDebugIdentifier);
        if (isDebugProperties && isTrueLiteral(assignmentStatementTree.assignedValue())) {
          ctx.addIssue(assignmentStatementTree, MESSAGE);
        }
      }
    });
  }

  private static boolean isDebugIdentifier(Expression expr) {
    return expr.is(Kind.NAME) && debugProperties.contains(((Name) expr).name());
  }

  private static boolean isTrueLiteral(Expression expr) {
    return expr.is(Kind.NAME) && ((Name) expr).name().equals("True");
  }

  private static boolean isDebugArgument(Argument argument) {
    Name keywordArgument = argument.is(Kind.REGULAR_ARGUMENT) ? ((RegularArgument) argument).keywordArgument() : null;
    if (keywordArgument != null && debugProperties.contains((keywordArgument).name())) {
      return isTrueLiteral(((RegularArgument) argument).expression());
    }
    return false;
  }

  @CheckForNull
  private static String getQualifiedName(CallExpression callExpression) {
    Symbol symbol = callExpression.calleeSymbol();
    return symbol != null ? symbol.fullyQualifiedName() : "";
  }
}
