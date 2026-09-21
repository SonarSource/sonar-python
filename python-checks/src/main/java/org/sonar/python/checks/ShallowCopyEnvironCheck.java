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

import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9158")
public class ShallowCopyEnvironCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this shallow copy of \"os.environ\" with \"os.environ.copy()\".";
  private static final String QUICK_FIX_MESSAGE = "Replace with \"os.environ.copy()\"";

  private static final TypeMatcher COPY_COPY_MATCHER = TypeMatchers.isType("copy.copy");
  private static final TypeMatcher OS_ENVIRON_MATCHER = TypeMatchers.isType("os.environ");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ShallowCopyEnvironCheck::checkCall);
  }

  private static void checkCall(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    if (!COPY_COPY_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }

    List<Argument> arguments = callExpression.arguments();
    if (arguments.size() != 1 || !(arguments.get(0) instanceof RegularArgument regularArgument)) {
      return;
    }
    if (!isCopyPositionalOrXKeyword(regularArgument)) {
      return;
    }

    Expression environExpression = Expressions.removeParentheses(regularArgument.expression());
    if (!OS_ENVIRON_MATCHER.isTrueFor(environExpression, ctx)) {
      return;
    }

    PreciseIssue issue = ctx.addIssue(callExpression, MESSAGE);
    createQuickFix(callExpression, environExpression).ifPresent(issue::addQuickFix);
  }

  private static boolean isCopyPositionalOrXKeyword(RegularArgument argument) {
    return argument.keywordArgument() == null || "x".equals(argument.keywordArgument().name());
  }

  private static Optional<PythonQuickFix> createQuickFix(CallExpression callExpression, Expression environExpression) {
    // Fixing multiline issues with comments in between is more trouble than its worth
    String environText = TreeUtils.treeToString(environExpression, false);
    if (environText == null || TreeUtils.treeToString(callExpression, false) == null) {
      return Optional.empty();
    }
    String replacement = environText + ".copy()";
    return Optional.of(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
      .addTextEdit(TextEditUtils.replace(callExpression, replacement))
      .build());
  }
}
