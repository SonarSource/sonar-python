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
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.checks.Expressions;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6786")
public class GraphQLIntrospectionCheck extends PythonSubscriptionCheck {

  private static final Set<String> GRAPHQL_VIEWS_FQNS = Set.of(
    "flask_graphql.GraphQLView",
    "graphql_server.flask.GraphQLView");

  private static final String AS_VIEW_CALLEE_NAME = "as_view";

  private static final Set<String> SAFE_VALIDATION_RULE_FQNS = Set.of(
    "graphene.validation.DisableIntrospection",
    "graphql.validation.NoSchemaIntrospectionCustomRule");

  private static final String MESSAGE = "Disable introspection on this \"GraphQL\" server endpoint.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, GraphQLIntrospectionCheck::checkGraphQLIntrospection);
  }

  private static void checkGraphQLIntrospection(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Optional.of(callExpression)
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .filter(GraphQLIntrospectionCheck::isCallToAsView)
      .map(QualifiedExpression::qualifier)
      .filter(HasSymbol.class::isInstance)
      .map(HasSymbol.class::cast)
      .map(HasSymbol::symbol)
      .filter(GraphQLIntrospectionCheck::isOrExtendsRelevantFqn)
      .filter(fqn -> !hasSafeMiddlewares(callExpression.arguments()))
      .filter(fqn -> !hasSafeValidationRules(callExpression.arguments()))
      .ifPresent(fqn -> ctx.addIssue(callExpression.callee(), MESSAGE));
  }

  private static boolean isOrExtendsRelevantFqn(Symbol symbol) {
    if (symbol.is(Symbol.Kind.CLASS)) {
      return GRAPHQL_VIEWS_FQNS.stream().anyMatch(((ClassSymbol) symbol)::isOrExtends);
    }
    return Optional.of(symbol)
      .map(Symbol::fullyQualifiedName)
      .filter(GRAPHQL_VIEWS_FQNS::contains)
      .isPresent();
  }

  private static boolean isCallToAsView(QualifiedExpression qualifiedExpression) {
    return Optional.of(qualifiedExpression)
      .map(QualifiedExpression::name)
      .map(Name::name)
      .filter(AS_VIEW_CALLEE_NAME::equals)
      .isPresent();
  }

  private static boolean hasSafeMiddlewares(List<Argument> arguments) {
    RegularArgument argument = TreeUtils.argumentByKeyword("middleware", arguments);
    if (argument == null) {
      return false;
    }

    return extractArgumentValues(argument)
      .map(values -> !values.isEmpty() && expressionsNameContainIntrospection(values))
      .orElse(true);
  }

  private static boolean hasSafeValidationRules(List<Argument> arguments) {
    RegularArgument argument = TreeUtils.argumentByKeyword("validation_rules", arguments);
    if (argument == null) {
      return false;
    }

    return extractArgumentValues(argument)
      .map(values -> !values.isEmpty() &&
        (expressionsNameContainIntrospection(values) || expressionsContainsSafeRuleFQN(values)))
      .orElse(true);
  }

  private static Optional<List<Expression>> extractArgumentValues(RegularArgument argument) {
    return Optional.of(argument)
      .map(RegularArgument::expression)
      .map(GraphQLIntrospectionCheck::ifNameGetSingleAssignedNonNameValue)
      .flatMap(GraphQLIntrospectionCheck::expressionsFromListOrTuple);
  }

  private static Expression ifNameGetSingleAssignedNonNameValue(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return Expressions.singleAssignedNonNameValue((Name) expression);
    }
    return expression;
  }

  private static Optional<List<Expression>> expressionsFromListOrTuple(Expression expression) {
    return TreeUtils.toOptionalInstanceOf(ListLiteral.class, expression)
      .map(ListLiteral::elements)
      .map(ExpressionList::expressions)
      .or(() -> TreeUtils.toOptionalInstanceOf(Tuple.class, expression)
        .map(Tuple::elements));
  }

  private static boolean expressionsNameContainIntrospection(List<Expression> expressions) {
    Stream<Optional<String>> expressionsNameAndType = Stream.concat(expressions.stream()
      .map(GraphQLIntrospectionCheck::nameFromIdentifierOrCallExpression),
      expressions.stream().map(GraphQLIntrospectionCheck::nameOfType));

    return expressionsNameAndType
      .filter(Optional::isPresent)
      .map(Optional::get)
      .map(String::toUpperCase)
      .anyMatch(name -> name.contains("INTROSPECTION"));
  }

  private static boolean expressionsContainsSafeRuleFQN(List<Expression> expressions) {
    return expressions.stream()
      .map(TreeUtils::getSymbolFromTree)
      .filter(Optional::isPresent)
      .map(Optional::get)
      .map(Symbol::fullyQualifiedName)
      .filter(Objects::nonNull)
      .anyMatch(SAFE_VALIDATION_RULE_FQNS::contains);
  }

  private static Optional<String> nameOfType(Expression expression) {
    return TreeUtils.toOptionalInstanceOf(NameImpl.class, expression)
      .map(NameImpl::type)
      .map(InferredType::runtimeTypeSymbol)
      .map(Symbol::name);
  }

  private static Optional<String> nameFromIdentifierOrCallExpression(Expression expression) {
    return Optional.ofNullable(TreeUtils.nameFromExpression(expression))
      .or(() -> TreeUtils.toOptionalInstanceOf(QualifiedExpression.class, expression)
        .map(TreeUtils::nameFromQualifiedExpression))
      .or(() -> TreeUtils.toOptionalInstanceOf(CallExpression.class, expression)
        .map(CallExpression::callee)
        .flatMap(GraphQLIntrospectionCheck::nameFromExpressionOrQualifiedExpression));
  }

  private static Optional<String> nameFromExpressionOrQualifiedExpression(Expression expression) {
    return TreeUtils.toOptionalInstanceOf(QualifiedExpression.class, expression)
      .map(TreeUtils::nameFromQualifiedExpression)
      .or(() -> Optional.ofNullable(TreeUtils.nameFromExpression(expression)));
  }
}
