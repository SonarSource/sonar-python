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
package org.sonar.python.checks;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6796")
public class GenericFunctionTypeParameterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a generic type parameter for this function instead of a \"TypeVar\".";
  private static final String SECONDARY_MESSAGE_USE = "Use of \"TypeVar\" here.";
  private static final String SECONDARY_MESSAGE_ASSIGNMENT = "\"TypeVar\" is assigned here.";
  public static final String TYPE_VAR_FQN = "typing.TypeVar";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, GenericFunctionTypeParameterCheck::checkUseOfGenerics);
  }

  private static void checkUseOfGenerics(SubscriptionContext ctx) {
    if (!supportsTypeParameterSyntax(ctx)) {
      return;
    }
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    Set<Tree> secondaryLocations = Optional.ofNullable(functionDef.parameters())
      .map(ParameterList::nonTuple)
      .stream()
      .flatMap(List::stream)
      .map(Parameter::typeAnnotation)
      .filter(Objects::nonNull)
      .map(TypeAnnotation::expression)
      .filter(GenericFunctionTypeParameterCheck::isGenericTypeAnnotation)
      .collect(Collectors.toSet());
    Optional.ofNullable(functionDef.returnTypeAnnotation())
      .map(TypeAnnotation::expression)
      .filter(GenericFunctionTypeParameterCheck::isGenericTypeAnnotation)
      .ifPresent(secondaryLocations::add);

    if (!secondaryLocations.isEmpty()) {
      PreciseIssue issue = ctx.addIssue(functionDef.name(), MESSAGE);
      secondaryLocations.forEach(loc -> issue.secondary(loc, SECONDARY_MESSAGE_USE));
      getAssignmentLocations(secondaryLocations).forEach(loc -> issue.secondary(loc, SECONDARY_MESSAGE_ASSIGNMENT));
    }
  }

  private static Set<Tree> getAssignmentLocations(Set<Tree> secondaryLocations) {
    return secondaryLocations.stream()
      .map(Name.class::cast)
      .map(Expressions::singleAssignedValue)
      .collect(Collectors.toSet());
  }

  private static boolean isGenericTypeAnnotation(Expression expression) {
    return Optional.of(expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Expressions::singleAssignedValue)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Name::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter(TYPE_VAR_FQN::equals)
      .isPresent();
  }

  private static boolean supportsTypeParameterSyntax(SubscriptionContext ctx) {
    PythonVersionUtils.Version required = PythonVersionUtils.Version.V_312;

    // All versions must be greater than or equal to the required version.
    return ctx.sourcePythonVersions().stream()
      .allMatch(version -> version.compare(required.major(), required.minor()) >= 0);
  }
}
