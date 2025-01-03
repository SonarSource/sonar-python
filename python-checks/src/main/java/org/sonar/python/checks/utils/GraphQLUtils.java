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
package org.sonar.python.checks.utils;

import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;

public class GraphQLUtils {

  private GraphQLUtils() {

  }

  private static final Set<String> GRAPHQL_VIEWS_FQNS = Set.of(
    "flask_graphql.GraphQLView",
    "graphql_server.flask.GraphQLView");

  private static final String AS_VIEW_CALLEE_NAME = "as_view";

  public static boolean isOrExtendsGraphQLView(Symbol symbol) {
    if (symbol.is(Symbol.Kind.CLASS)) {
      return GRAPHQL_VIEWS_FQNS.stream().anyMatch(((ClassSymbol) symbol)::isOrExtends);
    }
    return Optional.of(symbol)
      .map(Symbol::fullyQualifiedName)
      .filter(GRAPHQL_VIEWS_FQNS::contains)
      .isPresent();
  }

  public static boolean isCallToAsView(QualifiedExpression qualifiedExpression) {
    return Optional.of(qualifiedExpression)
      .map(QualifiedExpression::name)
      .map(Name::name)
      .filter(AS_VIEW_CALLEE_NAME::equals)
      .isPresent();
  }

  public static boolean expressionTypeOrNameMatchPredicate(Expression expression, Predicate<String> predicate) {
    Stream<Optional<String>> expressionNameAndType =
      Stream.of(TreeUtils.nameFromQualifiedOrCallExpression(expression), Optional.ofNullable(InferredTypes.typeName(expression.type())));

    return expressionNameAndType
      .filter(Optional::isPresent)
      .map(Optional::get)
      .anyMatch(predicate);
  }

  public static boolean expressionFQNMatchPredicate(Expression expression, Predicate<String> predicate) {
    return TreeUtils.getSymbolFromTree(expression)
      .map(Symbol::fullyQualifiedName)
      .filter(predicate)
      .isPresent();
  }
}
