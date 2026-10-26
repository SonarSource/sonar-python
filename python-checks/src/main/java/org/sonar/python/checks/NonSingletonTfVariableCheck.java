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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6918")
public class NonSingletonTfVariableCheck extends PythonSubscriptionCheck {
  private static final TypeMatcher TENSORFLOW_VARIABLE = TypeMatchers.isType("tensorflow.Variable");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, NonSingletonTfVariableCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    if (!TENSORFLOW_VARIABLE.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }
    if (!isWithinTensorflowFunction(callExpression)) {
      return;
    }
    if (isUnconditional(callExpression)) {
      ctx.addIssue(callExpression, "Refactor this variable declaration to be a singleton.");
    }
  }

  private static boolean isUnconditional(CallExpression callExpression) {
    Tree firstAncestor = TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.FUNCDEF, Tree.Kind.IF_STMT, Tree.Kind.CONDITIONAL_EXPR);
    return firstAncestor != null && firstAncestor.is(Tree.Kind.FUNCDEF);
  }

  private static boolean isWithinTensorflowFunction(Tree tree) {
    return TreeUtils.firstAncestor(tree, t -> TreeUtils.isFunctionWithGivenDecoratorFQN(t, "tensorflow.function")) != null;
  }
}
