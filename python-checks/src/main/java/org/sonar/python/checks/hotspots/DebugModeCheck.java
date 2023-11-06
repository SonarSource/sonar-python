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
import java.util.function.Predicate;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
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
  private static final String MESSAGE = "Make sure this debug feature is deactivated before delivering the code in production.";
  private static final List<String> debugProperties = Arrays.asList("DEBUG", "DEBUG_PROPAGATE_EXCEPTIONS");
  private static final List<String> settingFiles = Arrays.asList("global_settings.py", "settings.py");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, DebugModeCheck::callExpressionConsumer);
    context.registerSyntaxNodeConsumer(Kind.ASSIGNMENT_STMT, DebugModeCheck::assignmentStatementConsumer);
  }

  private static void callExpressionConsumer(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    List<Argument> arguments = callExpression.arguments();
    if (!(callExpression.callee() instanceof QualifiedExpression) || arguments.isEmpty()) {
      return;
    }

    if (DJANGO_CONFIGURE_FQN.equals(getQualifiedName(callExpression))) {
      arguments.stream().filter(DebugModeCheck::isDebugArgument).forEach(arg -> ctx.addIssue(arg, MESSAGE));
    }

    if (FLASK_RUN_FQN.equals(getQualifiedName(callExpression))) {
      RegularArgument debugArgument = TreeUtils.nthArgumentOrKeyword(2, "debug", arguments);
      Optional.ofNullable(debugArgument)
        .map(RegularArgument::expression)
        .filter(DebugModeCheck::isTrueLiteral)
        .ifPresent(name -> ctx.addIssue(debugArgument, MESSAGE));
    }
  }

  private static void assignmentStatementConsumer(SubscriptionContext ctx) {
    Optional.of(ctx.pythonFile().fileName())
      .filter((settingFiles::contains))
      .ifPresentOrElse(
        fileName -> assignmentStatementCheck(ctx, DebugModeCheck::hasDjangoOrFlaskDebugProperties),
        () -> assignmentStatementCheck(ctx, DebugModeCheck::hasFlaskDebugProperties));
  }

  private static void assignmentStatementCheck(SubscriptionContext ctx, Predicate<ExpressionList> isDebugProperty) {
    AssignmentStatement assignmentStatementTree = (AssignmentStatement) ctx.syntaxNode();
    for (ExpressionList lhsExpression : assignmentStatementTree.lhsExpressions()) {
      if (isDebugProperty.test(lhsExpression) && isTrueLiteral(assignmentStatementTree.assignedValue())) {
        ctx.addIssue(assignmentStatementTree, MESSAGE);
      }
    }
  }

  private static boolean hasDjangoOrFlaskDebugProperties(ExpressionList expressionList) {
    return expressionList.expressions().stream().anyMatch(DebugModeCheck::isDebugIdentifier) || hasFlaskDebugProperties(expressionList);
  }

  private static boolean hasFlaskDebugProperties(ExpressionList expressionList) {
    return expressionList.expressions().stream().anyMatch(DebugModeCheck::isModifyingFlaskDebugProperty);

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
      return isFlaskAppConfiguration(subscriptionExpression)
        && isMakingDebugParameterTrue(subscriptionExpression);
    }
    return false;
  }

  private static boolean isMakingDebugParameterTrue(SubscriptionExpression subscriptionExpression) {
    return Optional.of(subscriptionExpression.subscripts())
      .map(ExpressionList::expressions)
      .filter(list -> list.size() == 1)
      .map(list -> list.get(0))
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(StringLiteral.class))
      .map(StringLiteral::trimmedQuotesValue)
      .filter("DEBUG"::equals)
      .isPresent();
  }

  private static boolean isFlaskAppConfiguration(SubscriptionExpression subscriptionExpression) {
    return Optional.of(subscriptionExpression)
      .map(SubscriptionExpression::object)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(QualifiedExpression::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter(FLASK_APP_CONFIG_FQN::equals)
      .isPresent();
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
