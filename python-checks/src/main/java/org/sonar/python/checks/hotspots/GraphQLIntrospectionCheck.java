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

import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.GraphQLUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6786")
public class GraphQLIntrospectionCheck extends PythonSubscriptionCheck {

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
      .filter(GraphQLUtils::isCallToAsView)
      .map(QualifiedExpression::qualifier)
      .filter(HasSymbol.class::isInstance)
      .map(HasSymbol.class::cast)
      .map(HasSymbol::symbol)
      .filter(GraphQLUtils::isOrExtendsGraphQLView)
      .filter(fqn -> !hasSafeMiddlewares(callExpression.arguments()))
      .filter(fqn -> !hasSafeValidationRules(callExpression.arguments()))
      .ifPresent(fqn -> ctx.addIssue(callExpression.callee(), MESSAGE));
  }

  private static boolean hasSafeMiddlewares(List<Argument> arguments) {
    RegularArgument argument = TreeUtils.argumentByKeyword("middleware", arguments);
    if (argument == null) {
      return false;
    }

    Optional<Expression> argumentValue = Expressions.ifNameGetSingleAssignedNonNameValue(argument.expression());
    boolean isNotTupleNorListLiteral = argumentValue.filter(a -> a.is(Tree.Kind.LIST_LITERAL, Tree.Kind.TUPLE)).isEmpty();
    return isNotTupleNorListLiteral || Expressions.expressionsFromListOrTuple(argumentValue.get()).stream().anyMatch(GraphQLIntrospectionCheck::isSafeMiddlewareName);
  }

  private static boolean hasSafeValidationRules(List<Argument> arguments) {
    RegularArgument argument = TreeUtils.argumentByKeyword("validation_rules", arguments);
    if (argument == null) {
      return false;
    }

    Optional<Expression> argumentValue = Expressions.ifNameGetSingleAssignedNonNameValue(argument.expression());
    boolean isNotTupleNorListLiteral = argumentValue.filter(a -> a.is(Tree.Kind.LIST_LITERAL, Tree.Kind.TUPLE)).isEmpty();
    return isNotTupleNorListLiteral || Expressions.expressionsFromListOrTuple(argumentValue.get()).stream().anyMatch(GraphQLIntrospectionCheck::isSafeValidationRule);
  }

  private static boolean isSafeValidationRule(Expression value) {
    return isSafeMiddlewareName(value) || GraphQLUtils.expressionFQNMatchPredicate(value, SAFE_VALIDATION_RULE_FQNS::contains);
  }

  private static boolean isSafeMiddlewareName(Expression value) {
    return GraphQLUtils.expressionTypeOrNameMatchPredicate(value, name -> name.toUpperCase(Locale.ROOT).contains("INTROSPECTION"));
  }

}
