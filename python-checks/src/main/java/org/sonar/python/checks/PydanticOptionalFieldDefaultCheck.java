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
package org.sonar.python.checks;

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8396")
public class PydanticOptionalFieldDefaultCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add an explicit default value to this optional field.";

  private static final TypeMatcher IS_PYDANTIC_MODEL = TypeMatchers.isOrExtendsType("pydantic.BaseModel");
  private static final TypeMatcher IS_PYDANTIC_FIELD = TypeMatchers.isType("pydantic.Field");

  private static final TypeMatcher IS_TYPING_OPTIONAL = TypeMatchers.isType("typing.Optional");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, PydanticOptionalFieldDefaultCheck::checkClassDef);
  }

  private static void checkClassDef(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();

    if (!IS_PYDANTIC_MODEL.isTrueFor(classDef.name(), ctx)) {
      return;
    }

    classDef.body().statements().stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(AnnotatedAssignment.class))
      .forEach(annotatedAssignment -> checkField(ctx, annotatedAssignment));
  }

  private static void checkField(SubscriptionContext ctx, AnnotatedAssignment annotatedAssignment) {
    TypeAnnotation annotation = annotatedAssignment.annotation();
    Expression annotationExpr = annotation.expression();

    if (!isTypingOptional(annotationExpr, ctx)) {
      return;
    }

    Expression assignedValue = annotatedAssignment.assignedValue();
    if (assignedValue == null || isFieldCallWithEllipsis(ctx, assignedValue)) {
      ctx.addIssue(annotationExpr, MESSAGE);
    }
  }

  private static boolean isTypingOptional(Expression annotationExpr, SubscriptionContext ctx) {
    if (annotationExpr instanceof SubscriptionExpression subscriptionExpr) {
      return IS_TYPING_OPTIONAL.isTrueFor(subscriptionExpr.object(), ctx);
    }
    return false;
  }

  private static boolean isFieldCallWithEllipsis(SubscriptionContext ctx, Expression assignedValue) {
    if (!(assignedValue instanceof CallExpression callExpr)) {
      return false;
    }

    if (!IS_PYDANTIC_FIELD.isTrueFor(callExpr.callee(), ctx)) {
      return false;
    }

    return hasEllipsisAsFirstPositionalArg(callExpr) && !hasDefaultKeywordArg(callExpr);
  }

  private static boolean hasEllipsisAsFirstPositionalArg(CallExpression callExpr) {
    List<Argument> arguments = callExpr.arguments();
    if (arguments.isEmpty()) {
      return false;
    }

    Argument firstArg = arguments.get(0);
    if (firstArg instanceof RegularArgument regularArg && regularArg.keywordArgument() == null) {
      return regularArg.expression().is(Tree.Kind.ELLIPSIS);
    }
    return false;
  }

  private static boolean hasDefaultKeywordArg(CallExpression callExpr) {
    return callExpr.arguments().stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(RegularArgument.class))
      .anyMatch(arg -> {
        var keyword = arg.keywordArgument();
        return keyword != null && ("default".equals(keyword.name()) || "default_factory".equals(keyword.name()));
      });
  }

}
