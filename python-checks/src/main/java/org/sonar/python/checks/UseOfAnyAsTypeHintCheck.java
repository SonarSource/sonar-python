/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6542")
public class UseOfAnyAsTypeHintCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a more specific type than `Any` for this type hint.";
  private static final Set<String> OVERRIDE_FQNS = Set.of("typing.override", "typing.overload");
  private static final Set<String> OVERRIDE_NAMES = Set.of("override", "overload");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.RETURN_TYPE_ANNOTATION, UseOfAnyAsTypeHintCheck::checkForAnyInReturnTypeAndParameters);
    context.registerSyntaxNodeConsumer(Tree.Kind.PARAMETER_TYPE_ANNOTATION, UseOfAnyAsTypeHintCheck::checkForAnyInReturnTypeAndParameters);
    context.registerSyntaxNodeConsumer(Tree.Kind.VARIABLE_TYPE_ANNOTATION, UseOfAnyAsTypeHintCheck::checkForAnyInTypeHint);
  }

  private static void checkForAnyInTypeHint(SubscriptionContext ctx) {
    Optional.of((TypeAnnotation) ctx.syntaxNode())
      .filter(UseOfAnyAsTypeHintCheck::isTypeAny)
      .ifPresent(typeAnnotation -> ctx.addIssue(typeAnnotation.expression(), MESSAGE));
  }

  private static void checkForAnyInReturnTypeAndParameters(SubscriptionContext ctx) {
    TypeAnnotation typeAnnotation = (TypeAnnotation) ctx.syntaxNode();
    Optional.of(typeAnnotation)
      .filter(UseOfAnyAsTypeHintCheck::isTypeAny)
      .map(annotation -> (FunctionDef) TreeUtils.firstAncestorOfKind(annotation, Tree.Kind.FUNCDEF))
      .filter(Predicate.not(UseOfAnyAsTypeHintCheck::hasFunctionOverrideOrOverloadDecorator))
      .filter(Predicate.not(UseOfAnyAsTypeHintCheck::canFunctionBeAnOverride))
      .ifPresent(functionDef -> ctx.addIssue(typeAnnotation.expression(), MESSAGE));
  }

  private static boolean isTypeAny(@Nullable TypeAnnotation typeAnnotation) {
    return Optional.ofNullable(typeAnnotation)
      .map(TypeAnnotation::expression)
      .flatMap(TreeUtils::fullyQualifiedNameFromExpression)
      .map("typing.Any"::equals)
      .orElse(false);
  }

  private static boolean hasFunctionOverrideOrOverloadDecorator(FunctionDef currentFunctionDef) {
    return currentFunctionDef.decorators().stream()
      .map(Decorator::expression)
      .anyMatch(expression -> expression.is(Tree.Kind.NAME) && OVERRIDE_NAMES.contains(((Name) expression).name()))
      ||
      currentFunctionDef.decorators().stream()
        .map(Decorator::expression)
        .map(TreeUtils::fullyQualifiedNameFromExpression)
        .flatMap(Optional::stream)
        .anyMatch(OVERRIDE_FQNS::contains);
  }

  private static boolean canFunctionBeAnOverride(FunctionDef currentMethodDef) {
    return Optional.ofNullable(TreeUtils.getFunctionSymbolFromDef(currentMethodDef))
      .map(SymbolUtils::canBeAnOverridingMethod)
      .orElse(false);
  }

}
