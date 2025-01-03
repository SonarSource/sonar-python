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

import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;


public abstract class FlaskHardCodedSecret extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Don't disclose %s secret keys.";
  private static final String SECONDARY_MESSAGE = "Assignment to sensitive property.";
  private static final Set<String> FLASK_APP_CONFIG_QUALIFIER_FQNS = Set.of(
    "flask.app.Flask.config",
    "flask.globals.current_app.config"
  );
  public static final String SECONDARY_LOCATION_MESSAGE = "The secret is used in this call.";

  protected abstract String getSecretKeyKeyword();

  protected abstract String getSecretKeyType();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::verifyCallExpression);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, this::verifyAssignmentStatement);
  }

  private void verifyCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Optional.of(callExpression)
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .filter(qualiExpr -> "update".equals(qualiExpr.name().name()))
      .map(QualifiedExpression::qualifier)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(QualifiedExpression::name)
      .map(Name::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter(FLASK_APP_CONFIG_QUALIFIER_FQNS::contains)
      .ifPresent(fqn -> verifyUpdateCallArgument(ctx, callExpression));
  }

  private void verifyUpdateCallArgument(SubscriptionContext ctx, CallExpression callExpression) {
    Optional.of(callExpression.arguments())
      .filter(arguments -> arguments.size() == 1)
      .map(arguments -> arguments.get(0))
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(RegularArgument.class))
      .map(RegularArgument::expression)
      .map(FlaskHardCodedSecret::getAssignedValue)
      .flatMap(this::getIllegalDictArgument)
      .ifPresent(illegalArgument -> ctx.addIssue(illegalArgument, getMessage())
        .secondary(callExpression.callee(), SECONDARY_LOCATION_MESSAGE));
  }

  private String getMessage() {
    return String.format(MESSAGE, getSecretKeyType());
  }

  private static Expression getAssignedValue(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return Expressions.singleAssignedValue((Name) expression);
    }
    return expression;
  }

  private Optional<Tree> getIllegalDictArgument(Expression expression) {
    if (expression.is(Tree.Kind.CALL_EXPR)) {
      return TreeUtils.toOptionalInstanceOf(CallExpression.class, expression)
        .filter(FlaskHardCodedSecret::isCallToDictConstructor)
        .flatMap(this::getIllegalKeywordArgument);
    } else if (expression.is(Tree.Kind.DICTIONARY_LITERAL)) {
      return TreeUtils.toOptionalInstanceOf(DictionaryLiteral.class, expression)
        .flatMap(this::getIllegalKeyValuePair);
    }
    return Optional.empty();
  }

  private static boolean isCallToDictConstructor(CallExpression callExpression) {
    return Optional.of(callExpression)
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Name::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter("dict"::equals)
      .isPresent();
  }

  private Optional<KeyValuePair> getIllegalKeyValuePair(DictionaryLiteral dictionaryLiteral) {
    return dictionaryLiteral.elements().stream()
      .filter(KeyValuePair.class::isInstance)
      .map(KeyValuePair.class::cast)
      .filter(this::isIllegalKeyValuePair)
      .findFirst();
  }

  private boolean isIllegalKeyValuePair(KeyValuePair keyValuePair) {
    return Optional.of(keyValuePair.key())
      .filter(StringLiteral.class::isInstance)
      .map(StringLiteral.class::cast)
      .map(StringLiteral::trimmedQuotesValue)
      .filter(getSecretKeyKeyword()::equals)
      .isPresent() && isStringValue(keyValuePair.value());
  }

  private Optional<RegularArgument> getIllegalKeywordArgument(CallExpression callExpression) {
    return Optional.ofNullable(TreeUtils.argumentByKeyword(getSecretKeyKeyword(), callExpression.arguments()))
      .filter(argument -> Optional.of(argument)
        .map(RegularArgument::expression)
        .filter(FlaskHardCodedSecret::isStringValue)
        .isPresent());
  }

  private void verifyAssignmentStatement(SubscriptionContext ctx) {
    AssignmentStatement assignmentStatementTree = (AssignmentStatement) ctx.syntaxNode();
    if (!isStringValue(assignmentStatementTree.assignedValue())) {
      return;
    }
    List<Expression> expressionList = assignmentStatementTree.lhsExpressions().stream()
      .map(ExpressionList::expressions)
      .flatMap(List::stream)
      .filter(this::isSensitiveProperty)
      .toList();
    if (!expressionList.isEmpty()) {
      PreciseIssue issue = ctx.addIssue(assignmentStatementTree.assignedValue(), getMessage());
      expressionList.forEach(expr -> issue.secondary(expr, SECONDARY_MESSAGE));
    }
  }

  protected boolean isSensitiveProperty(Expression expression) {
    if (!expression.is(Tree.Kind.SUBSCRIPTION)) {
      return false;
    }
    return Optional.of((SubscriptionExpression) expression)
      .map(SubscriptionExpression::object)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(QualifiedExpression::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter(FLASK_APP_CONFIG_QUALIFIER_FQNS::contains)
      .map(fqn -> ((SubscriptionExpression) expression).subscripts())
      .map(ExpressionList::expressions)
      .filter(list -> list.size() == 1)
      .map(list -> list.get(0))
      .map(FlaskHardCodedSecret::getAssignedValue)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(StringLiteral.class))
      .map(StringLiteral::trimmedQuotesValue)
      .filter(getSecretKeyKeyword()::equals)
      .isPresent();
  }

  private static boolean isStringValue(@Nullable Expression expr) {
    return isStringValue(expr, new HashSet<>());
  }


  private static boolean isStringValue(@Nullable Expression expr, Set<String> visited) {
    if (expr == null) {
      return false;
    }
    if (expr.is(Tree.Kind.NAME)) {
      if (visited.contains(((Name) expr).name())) {
        return false;
      }
      visited.add(((Name) expr).name());
      Expression assignmentValueExpression = Expressions.singleAssignedValue((Name) expr);
      return isStringValue(assignmentValueExpression, visited);
    } else {
      return expr.is(Tree.Kind.STRING_LITERAL);
    }
  }
}
