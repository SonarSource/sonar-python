/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
import java.util.Optional;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.tree.TreeUtils;

@Rule(key = DebugModeCheck.CHECK_KEY)
public class DebugModeCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S4507";
  public static final String DJANGO_CONFIGURE_FQN = "django.conf.settings.configure";
  private static final String FLASK_RUN_FQN = "flask.app.Flask.run";
  private static final String FLASK_APP_CONFIG_FQN = "flask.app.Flask.config";
  public static final String FLASK_APP_DEBUG_FQN = "flask.app.Flask.debug";
  public static final String TRUE_KEYWORD = "True";

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

//      LOG.info(getQualifiedName(callExpression));
      if (DJANGO_CONFIGURE_FQN.equals(getQualifiedName(callExpression)) && !arguments.isEmpty()) {
        arguments.stream().filter(DebugModeCheck::isDebugArgument).forEach(arg -> ctx.addIssue(arg, MESSAGE));
      }

      if (FLASK_RUN_FQN.equals(getQualifiedName(callExpression)) && !arguments.isEmpty()) {
        RegularArgument debugArgument = TreeUtils.nthArgumentOrKeyword(2, "debug", arguments);
        Optional.ofNullable(debugArgument)
          .map(RegularArgument::expression)
          .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
          .map(Name::name)
          .filter(TRUE_KEYWORD::equals)
          .ifPresent(str -> ctx.addIssue(debugArgument, MESSAGE));
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

    context.registerSyntaxNodeConsumer(Kind.ASSIGNMENT_STMT, ctx -> {
      AssignmentStatement assignmentStatementTree = (AssignmentStatement) ctx.syntaxNode();
      for (ExpressionList lhsExpression : assignmentStatementTree.lhsExpressions()) {
        boolean isDebugProperties = lhsExpression.expressions().stream().anyMatch(DebugModeCheck::isModifyingFlaskDebugProperty);
        if (isDebugProperties && isTrueLiteral(assignmentStatementTree.assignedValue())) {
          ctx.addIssue(assignmentStatementTree, MESSAGE);
        }
      }
    });

  }

  private static boolean isModifyingFlaskDebugProperty(Expression expression) {
    if (expression.is(Kind.QUALIFIED_EXPR)) {
      return Optional.of((QualifiedExpression) expression)
        .map(QualifiedExpression::symbol)
        .map(Symbol::fullyQualifiedName)
        .filter(FLASK_APP_DEBUG_FQN::equals)
        .isPresent();
    } else if (expression.is(Kind.SUBSCRIPTION)) {
      SubscriptionExpression subscriptionExpression = (SubscriptionExpression) expression;
      return Optional.of(subscriptionExpression)
        .map(SubscriptionExpression::object)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
        .map(QualifiedExpression::symbol)
        .map(Symbol::fullyQualifiedName)
        .filter(FLASK_APP_CONFIG_FQN::equals)
        .isPresent() && Optional.of(subscriptionExpression.subscripts())
        .map(ExpressionList::expressions)
        .filter(list -> list.size() == 1)
        .map(list -> list.get(0))
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(StringLiteral.class))
        .map(StringLiteral::trimmedQuotesValue)
        .filter("DEBUG"::equals)
        .isPresent();
    }
    return false;
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
