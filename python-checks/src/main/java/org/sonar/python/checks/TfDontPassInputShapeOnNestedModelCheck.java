/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6919")
public class TfDontPassInputShapeOnNestedModelCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Remove this `input_shape` argument, it is deprecated.";
  public static final String ARGUMENT_NAME = "input_shape";
  public static final Set<String> CLASS_FQN = Set.of("tensorflow.keras.Model", "tf.keras.Model", "keras.Model");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, TfDontPassInputShapeOnNestedModelCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext context) {
    if (Optional.of((CallExpression) context.syntaxNode())
      .filter(TfDontPassInputShapeOnNestedModelCheck::isSuperInitCall)
      .map(callExpression -> TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.FUNCDEF))
      .map(FunctionDef.class::cast)
      .filter(functionDef -> "__init__".equals(functionDef.name().name()))
      .map(funcDef -> TreeUtils.firstAncestorOfKind(funcDef, Tree.Kind.CLASSDEF))
      .map(ClassDef.class::cast)
      .map(TreeUtils::getClassSymbolFromDef)
      .filter(
        classSymbol -> classSymbol.superClasses().stream().anyMatch(superClass -> superClass.fullyQualifiedName() != null && CLASS_FQN.contains(superClass.fullyQualifiedName())))
      .isEmpty()) {
      return;
    }

    RegularArgument inputShapeArgument = TreeUtils.nthArgumentOrKeyword(-1, ARGUMENT_NAME, ((CallExpression) context.syntaxNode()).arguments());
    if (inputShapeArgument != null) {
      context.addIssue(inputShapeArgument, MESSAGE);
    }
  }

  private static boolean isSuperInitCall(CallExpression callExpression) {
    Expression callee = callExpression.callee();
    if (callee.is(Tree.Kind.QUALIFIED_EXPR)) {
      QualifiedExpression qualifiedExpression = (QualifiedExpression) callee;
      return "__init__".equals(qualifiedExpression.name().name()) && Optional.of(qualifiedExpression.qualifier())
        .filter(expr -> expr.is(Tree.Kind.CALL_EXPR))
        .map(CallExpression.class::cast)
        .map(CallExpression::calleeSymbol)
        .filter(symbol -> symbol.kind() == Symbol.Kind.CLASS)
        .map(Symbol::fullyQualifiedName)
        .filter("super"::equals)
        .isPresent();
    }
    return false;
  }
}
