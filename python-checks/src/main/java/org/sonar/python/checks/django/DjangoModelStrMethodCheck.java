/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.checks.django;

import java.util.Objects;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.django.DjangoUtils.getFieldAssignment;
import static org.sonar.python.checks.django.DjangoUtils.getMetaClass;

@Rule(key = "S6554")
public class DjangoModelStrMethodCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Define a \"__str__\" method for this Django model.";
  private static final TypeMatcher IS_DJANGO_MODEL = TypeMatchers.isType("django.db.models.base.Model");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      var classDef = (ClassDef) ctx.syntaxNode();
      if (isDirectDjangoModelSubclass(classDef, ctx)) {
        if (isAbstractModel(classDef)) {
          return;
        }
        boolean hasStrMethod = hasStrMethod(classDef);
        if (!hasStrMethod) {
          ctx.addIssue(classDef.name(), MESSAGE);
        }
      }

    });
  }

  private static boolean isDirectDjangoModelSubclass(ClassDef classDef, SubscriptionContext ctx) {
    var args = classDef.args();
    if (args == null) {
      return false;
    }
    return args.arguments().stream()
      .filter(RegularArgument.class::isInstance)
      .map(RegularArgument.class::cast)
      .map(RegularArgument::expression)
      .anyMatch(expr -> IS_DJANGO_MODEL.isTrueFor(expr, ctx));
  }

  private static boolean isAbstractModel(ClassDef classDef) {
    return getMetaClass(classDef)
      .flatMap(metaClass -> getFieldAssignment(metaClass, "abstract"))
      .filter(assignmentStatement -> Expressions.isTruthy(assignmentStatement.assignedValue()))
      .isPresent();
  }

  private static boolean hasStrMethod(ClassDef classDef) {
    var strMethodFqn = getStrMethodFqn(classDef);
    return TreeUtils.topLevelFunctionDefs(classDef)
      .stream()
      .map(TreeUtils::getFunctionSymbolFromDef)
      .filter(Objects::nonNull)
      .map(Symbol::fullyQualifiedName)
      .filter(Objects::nonNull)
      .anyMatch(strMethodFqn::equals);
  }

  private static String getStrMethodFqn(ClassDef classDef) {
    var classFqn = Optional.of(classDef)
      .map(TreeUtils::getClassSymbolFromDef)
      .map(Symbol::fullyQualifiedName)
      .orElse("");
    return classFqn + ".__str__";
  }
}
