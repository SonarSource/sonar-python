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

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6792")
public class GenericClassTypeParameterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use the type parameter syntax to declare this generic class.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, GenericClassTypeParameterCheck::checkGenericInheritance);
  }

  private static void checkGenericInheritance(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();

    ArgList args = classDef.args();
    if (args == null) {
      return;
    }
    args.arguments().stream()
      .map(GenericClassTypeParameterCheck::checkGenericValue)
      .filter(lst -> !lst.isEmpty())
      .findFirst()
      .ifPresent(locations -> {
        PreciseIssue issue = ctx.addIssue(classDef.name(), MESSAGE);
        locations.forEach(issue::secondary);
      });
  }

  private static List<IssueLocation> checkGenericValue(Argument argument) {
    if (!argument.is(Tree.Kind.REGULAR_ARGUMENT)) {
      return Collections.emptyList();
    }
    Expression expression = ((RegularArgument) argument).expression();

    if (isTypingGeneric(expression)) {
      return List.of(IssueLocation.preciseLocation(argument, "TODO"));
    }
    if (expression.is(Tree.Kind.NAME)) {
      Name name = (Name) expression;
      Expression assignedValue = Expressions.singleAssignedValue(name);
      if (assignedValue != null && isTypingGeneric(assignedValue)) {
        return List.of(
          IssueLocation.preciseLocation(argument, "TODO"),
          IssueLocation.preciseLocation(assignedValue, "TODO"));
      }
    }
    return Collections.emptyList();
  }

  private static boolean isTypingGeneric(Expression expression) {
    if (expression.is(Tree.Kind.SUBSCRIPTION)) {
      expression = ((SubscriptionExpression) expression).object();
    }
    return Optional.of(expression)
      .filter(HasSymbol.class::isInstance)
      .map(HasSymbol.class::cast)
      .map(HasSymbol::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter("typing.Generic"::equals)
      .isPresent();

  }
}
