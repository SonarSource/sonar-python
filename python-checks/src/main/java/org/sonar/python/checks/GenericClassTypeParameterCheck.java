/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S6792")
public class GenericClassTypeParameterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use the \"type\" parameter syntax to declare this generic class.";
  private static final String SECONDARY_MESSAGE_PARENT = "\"Generic\" parent.";
  private static final String SECONDARY_MESSAGE_ASSIGNMENT = "\"Generic\" is assigned here.";
  private static final TypeMatcher TYPING_GENERIC_MATCHER = TypeMatchers.isType("typing.Generic");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, GenericClassTypeParameterCheck::checkGenericInheritance);
  }

  private static void checkGenericInheritance(SubscriptionContext ctx) {
    if (!supportsTypeParameterSyntax(ctx)) {
      return;
    }
    ClassDef classDef = (ClassDef) ctx.syntaxNode();
    Optional.of(classDef)
      .map(ClassDef::args)
      .map(ArgList::arguments)
      .stream()
      .flatMap(List::stream)
      .map(argument -> checkGenericValue(argument, ctx))
      .filter(lst -> !lst.isEmpty())
      .findFirst()
      .ifPresent(locations -> {
        PreciseIssue issue = ctx.addIssue(classDef.name(), MESSAGE);
        locations.forEach(issue::secondary);
      });
  }

  private static List<IssueLocation> checkGenericValue(Argument argument, SubscriptionContext ctx) {
    if (!argument.is(Tree.Kind.REGULAR_ARGUMENT)) {
      return Collections.emptyList();
    }
    Expression expression = ((RegularArgument) argument).expression();

    if (expression.is(Tree.Kind.NAME)) {
      Name name = (Name) expression;
      Expression assignedValue = singleAssignedValue(name);
      if (assignedValue != null && isTypingGeneric(assignedValue, ctx)) {
        return List.of(
          IssueLocation.preciseLocation(argument, SECONDARY_MESSAGE_PARENT),
          IssueLocation.preciseLocation(assignedValue, SECONDARY_MESSAGE_ASSIGNMENT));
      }
    }
    if (isTypingGeneric(expression, ctx)) {
      return List.of(IssueLocation.preciseLocation(argument, SECONDARY_MESSAGE_PARENT));
    }
    return Collections.emptyList();
  }

  private static boolean isTypingGeneric(Expression expression, SubscriptionContext ctx) {
    if (expression.is(Tree.Kind.SUBSCRIPTION)) {
      expression = ((SubscriptionExpression) expression).object();
    }
    return TYPING_GENERIC_MATCHER.isTrueFor(expression, ctx);
  }

  private static Expression singleAssignedValue(Name name) {
    return Optional.ofNullable(name.symbolV2())
      .flatMap(SymbolV2::getSingleBindingUsage)
      .filter(usage -> usage.kind() == UsageV2.Kind.ASSIGNMENT_LHS)
      .map(UsageV2::tree)
      .map(Tree::parent)
      .map(GenericClassTypeParameterCheck::assignedValue)
      .orElse(null);
  }

  private static Expression assignedValue(Tree parent) {
    if (parent instanceof AnnotatedAssignment annotatedAssignment) {
      return annotatedAssignment.assignedValue();
    }
    if (parent instanceof ExpressionList expressionList
      && expressionList.expressions().size() == 1
      && parent.parent() instanceof AssignmentStatement assignment) {
      return assignment.assignedValue();
    }
    return null;
  }

  private static boolean supportsTypeParameterSyntax(SubscriptionContext ctx) {
    return PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(ctx.sourcePythonVersions(), PythonVersionUtils.Version.V_312);
  }
}
