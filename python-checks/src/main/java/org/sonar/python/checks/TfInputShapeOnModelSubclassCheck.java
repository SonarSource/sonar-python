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

import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6919")
public class TfInputShapeOnModelSubclassCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Remove this `input_shape` argument, it is deprecated.";
  public static final String ARGUMENT_NAME = "input_shape";
  public static final Set<String> CLASS_FQN = Set.of("tensorflow.keras.Model", "tensorflow.keras.models.Model", "tf.keras.Model", "keras.Model");

  private static final TypeMatcher MODEL_BASE = TypeMatchers.any(CLASS_FQN.stream()
    .map(fqn -> TypeMatchers.any(TypeMatchers.isType(fqn), TypeMatchers.withFQN(fqn))).toList());
  private static final TypeMatcher SUPER = TypeMatchers.isType("builtins.super");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, TfInputShapeOnModelSubclassCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    if (!isSuperInitCall(callExpression, context)) {
      return;
    }

    if (!isWithinInitOfTfModel(callExpression, context)) {
      return;
    }

    RegularArgument inputShapeArgument = TreeUtils.nthArgumentOrKeyword(-1, ARGUMENT_NAME, ((CallExpression) context.syntaxNode()).arguments());
    if (inputShapeArgument != null) {
      context.addIssue(inputShapeArgument, MESSAGE);
    }
  }

  private static boolean isWithinInitOfTfModel(CallExpression callExpression, SubscriptionContext context) {
    return Optional.of(callExpression)
      .map(c -> TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.FUNCDEF))
      .map(FunctionDef.class::cast)
      .filter(functionDef -> "__init__".equals(functionDef.name().name()))
      .map(funcDef -> TreeUtils.firstAncestorOfKind(funcDef, Tree.Kind.CLASSDEF))
      .map(ClassDef.class::cast)
      .map(ClassDef::args)
      .filter(args -> args.arguments().stream()
        .flatMap(TreeUtils.toStreamInstanceOfMapper(RegularArgument.class))
        .filter(argument -> argument.keywordArgument() == null)
        .anyMatch(argument -> MODEL_BASE.isTrueFor(argument.expression(), context)))
      .isPresent();
  }

  private static boolean isSuperInitCall(CallExpression callExpression, SubscriptionContext context) {
    Expression callee = callExpression.callee();
    if (!callee.is(Tree.Kind.QUALIFIED_EXPR)) {
      return false;
    }
    QualifiedExpression qualifiedExpression = (QualifiedExpression) callee;
    return "__init__".equals(qualifiedExpression.name().name()) && isCallToSuper(qualifiedExpression, context);
  }

  private static boolean isCallToSuper(QualifiedExpression qualifiedExpression, SubscriptionContext context) {
    return Optional.of(qualifiedExpression.qualifier())
      .filter(expr -> expr.is(Tree.Kind.CALL_EXPR))
      .map(CallExpression.class::cast)
      .map(CallExpression::callee)
      .filter(callee -> SUPER.isTrueFor(callee, context))
      .isPresent();
  }
}
