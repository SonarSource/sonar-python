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
package org.sonar.python.checks.utils;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
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

  public static Optional<List<Expression>> extractListOrTupleArgumentValues(RegularArgument argument) {
    return Optional.of(argument)
      .map(RegularArgument::expression)
      .map(Expressions::ifNameGetSingleAssignedNonNameValue)
      .flatMap(Expressions::expressionsFromListOrTuple);
  }

  public static boolean expressionsNameMatchPredicate(List<Expression> expressions, Predicate<String> predicate) {
    Stream<Optional<String>> expressionsNameAndType = Stream.concat(expressions.stream()
        .map(TreeUtils::nameFromQualifiedOrCallExpression),
      expressions.stream().map(Expression::type).map(t -> Optional.ofNullable(InferredTypes.typeName(t))));

    return expressionsNameAndType
      .filter(Optional::isPresent)
      .map(Optional::get)
      .anyMatch(predicate);
  }

  public static boolean expressionsContainsSafeRuleFQN(List<Expression> expressions, Predicate<String> predicate) {
    return expressions.stream()
      .map(TreeUtils::getSymbolFromTree)
      .filter(Optional::isPresent)
      .map(Optional::get)
      .map(Symbol::fullyQualifiedName)
      .filter(Objects::nonNull)
      .anyMatch(predicate);
  }
}
