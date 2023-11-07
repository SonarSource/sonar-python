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

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6786")
public class GraphQLIntrospectionCheck extends PythonSubscriptionCheck {

  private static final Set<String> GRAPHQL_VIEWS_FQNS = Set.of(
    "flask_graphql.GraphQLView.as_view",
    "graphql_server.flask.GraphQLView.as_view");

  private static final Set<String> SENSITIVE_VALIDATION_RULE_FQNS = Set.of(
    "graphene.validation.DisableIntrospection",
    "graphql.validation.NoSchemaIntrospectionCustomRule");

  private static final String MESSAGE = "Disable introspection on this GraphQL server endpoint.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, GraphQLIntrospectionCheck::checkGraphQLIntrospection);
  }

  private static void checkGraphQLIntrospection(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(callExpression.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(fqn -> GRAPHQL_VIEWS_FQNS.contains(fqn) &&
        (hasSensitiveMiddlewares(callExpression.arguments()) || hasSensitiveValidationRules(callExpression.arguments())))
      .ifPresent(fqn -> ctx.addIssue(callExpression, MESSAGE));
  }

  private static boolean hasSensitiveMiddlewares(List<Argument> arguments) {
    return areArgumentsUnsafe("middleware", expressionsNameContainIntrospection, arguments);
  }

  private static boolean hasSensitiveValidationRules(List<Argument> arguments) {
    return areArgumentsUnsafe("validation_rules",
      expressionsNameContainIntrospection.or(expressionsContainsSensitiveRuleFQN),
      arguments);
  }

  private static boolean areArgumentsUnsafe(String argumentName, Predicate<List<Expression>> check, List<Argument> arguments) {
    var middlewareArgument = TreeUtils.argumentByKeyword(argumentName, arguments);
    if (middlewareArgument == null) {
      return true;
    }
    return Optional.of(middlewareArgument)
      .map(RegularArgument::expression)
      .flatMap(GraphQLIntrospectionCheck::expressionsFromListOrTuple)
      .map(expressions -> expressions.isEmpty() || check.test(expressions))
      .orElse(false);
  }

  private static Optional<List<Expression>> expressionsFromListOrTuple(Expression expression) {
    return TreeUtils.toOptionalInstanceOfMapper(ListLiteral.class).apply(expression)
      .map(ListLiteral::elements)
      .map(ExpressionList::expressions)
      .or(() -> TreeUtils.toOptionalInstanceOfMapper(Tuple.class)
        .apply(expression)
        .map(Tuple::elements));
  }

  private static Predicate<List<Expression>> expressionsNameContainIntrospection = expressions -> expressions.stream()
    .map(GraphQLIntrospectionCheck::nameFromIndentifierOrCallExpression)
    .filter(Optional::isPresent)
    .map(Optional::get)
    .map(String::toUpperCase)
    .anyMatch(name -> name.contains("INTROSPECTION"));

  private static Predicate<List<Expression>> expressionsContainsSensitiveRuleFQN = expressions -> expressions.stream()
    .map(TreeUtils::getSymbolFromTree)
    .filter(Optional::isPresent)
    .map(Optional::get)
    .map(Symbol::fullyQualifiedName)
    .filter(Objects::nonNull)
    .anyMatch(SENSITIVE_VALIDATION_RULE_FQNS::contains);

  private static Optional<String> nameFromIndentifierOrCallExpression(Expression expression) {
    return Optional.ofNullable(TreeUtils.nameFromExpression(expression))
      .or(() -> TreeUtils.toOptionalInstanceOfMapper(CallExpression.class)
        .apply(expression)
        .map(CallExpression::callee)
        .map(TreeUtils::nameFromExpression))
      .filter(Objects::nonNull);
  }
}
