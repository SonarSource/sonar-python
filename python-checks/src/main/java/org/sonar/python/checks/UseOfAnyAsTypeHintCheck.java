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
import java.util.Optional;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6542")
public class UseOfAnyAsTypeHintCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a more specific type than `Any` for this type hint.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.RETURN_TYPE_ANNOTATION, UseOfAnyAsTypeHintCheck::checkForAnyInTypeHint);
    context.registerSyntaxNodeConsumer(Tree.Kind.PARAMETER_TYPE_ANNOTATION, UseOfAnyAsTypeHintCheck::checkForAnyInTypeHint);
    context.registerSyntaxNodeConsumer(Tree.Kind.VARIABLE_TYPE_ANNOTATION, UseOfAnyAsTypeHintCheck::checkForAnyInTypeHint);
  }

  private static void checkForAnyInTypeHint(SubscriptionContext ctx) {
    TypeAnnotation typeAnnotation = (TypeAnnotation) ctx.syntaxNode();
    Optional.of(typeAnnotation)
      .filter(UseOfAnyAsTypeHintCheck::isTypeAny)
      .filter(Predicate.not(UseOfAnyAsTypeHintCheck::isAnnotatingArgumentToChildMethod))
      .filter(Predicate.not(UseOfAnyAsTypeHintCheck::isUnannotatedOverride))
      .ifPresent(typeAnno -> ctx.addIssue(typeAnnotation.expression(), MESSAGE));
  }

  private static boolean isTypeAny(@Nullable TypeAnnotation typeAnnotation) {
    return Optional.ofNullable(typeAnnotation)
      .map(annotation -> "typing.Any".equals(TreeUtils.fullyQualifiedNameFromExpression(annotation.expression())))
      .orElse(false);
  }

  private static boolean isAnnotatingArgumentToChildMethod(TypeAnnotation typeAnnotation) {
    FunctionDef parentFunctionDef = (FunctionDef) TreeUtils.firstAncestorOfKind(typeAnnotation, Tree.Kind.FUNCDEF);
    if (parentFunctionDef == null) {
      return false;
    }
    return parentFunctionDef.decorators().stream()
      .map(Decorator::expression)
      .map(exp -> TreeUtils.toOptionalInstanceOf(Name.class, exp))
      .filter(Optional::isPresent)
      .map(Optional::get)
      .map(Name::name)
      .map(Pattern.compile("over(ride|load)")::matcher)
      .anyMatch(Matcher::matches);
  }

  private static boolean isUnannotatedOverride(TypeAnnotation typeAnnotation) {
    FunctionDef parentFunctionDef = (FunctionDef) TreeUtils.firstAncestorOfKind(typeAnnotation, Tree.Kind.FUNCDEF);
    if (parentFunctionDef == null) {
      return false;
    }
    String name = parentFunctionDef.name().name();
    return Optional.of(parentFunctionDef)
      .map(functionDef -> TreeUtils.firstAncestorOfKind(functionDef, Tree.Kind.CLASSDEF))
      .map(ClassDef.class::cast)
      .map(TreeUtils::getClassSymbolFromDef)
      .map(ClassSymbol::superClasses)
      .filter(list -> hasMemberOfName(name, list))
      .isPresent();
  }

  private static boolean hasMemberOfName(String name, List<Symbol> symbolList) {
    return symbolList.stream().filter(ClassSymbol.class::isInstance)
      .map(ClassSymbol.class::cast)
      .anyMatch(classSymbol -> classSymbol.canHaveMember(name));
  }
}
