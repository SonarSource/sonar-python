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
package org.sonar.python.checks.tests;

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.MockPatchUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9137")
public class MocksShouldUseAutospecCheck extends PythonSubscriptionCheck {

  private static final String MOCK_MESSAGE_FORMAT =
    "Replace this \"%s()\" with \"create_autospec(<collaborator>)\", or pass \"spec=\" / \"spec_set=\".";
  private static final String PATCH_MESSAGE =
    "Add \"autospec=True\" to this patch call, or pass an explicit \"spec=\" / \"spec_set=\".";
  private static final String AUTOSPEC_FALSE_MESSAGE =
    "Replace \"autospec=False\" with \"autospec=True\", or pass an explicit \"spec=\" / \"spec_set=\".";

  private static final String SPEC = "spec";
  private static final String SPEC_SET = "spec_set";
  private static final String AUTOSPEC = "autospec";
  private static final String NEW = "new";
  private static final String NEW_CALLABLE = "new_callable";

  private static final TypeMatcher MOCK_CLASS_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("unittest.mock.Mock"),
    TypeMatchers.isType("unittest.mock.MagicMock"),
    TypeMatchers.isType("mock.mock.Mock"),
    TypeMatchers.isType("mock.mock.MagicMock"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, MocksShouldUseAutospecCheck::checkCall);
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkCall(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    List<Argument> arguments = callExpression.arguments();
    if (Expressions.containsSpreadOperator(arguments)) {
      return;
    }

    Expression callee = Expressions.removeParentheses(callExpression.callee());
    if (MOCK_CLASS_MATCHER.isTrueFor(callee, ctx)) {
      if (!hasMockSpec(arguments)) {
        ctx.addIssue(callee, mockMessage(callee));
      }
      return;
    }

    int newPosition = MockPatchUtils.newArgumentPosition(callee, ctx);
    if (newPosition < 0) {
      return;
    }
    if (hasReplacement(arguments, newPosition) || hasPatchSpecOrAutospec(arguments, newPosition)) {
      return;
    }
    ctx.addIssue(callee, patchMessage(arguments, newPosition));
  }

  private static String mockMessage(Expression callee) {
    return String.format(MOCK_MESSAGE_FORMAT, mockConstructorName(callee));
  }

  private static String mockConstructorName(Expression callee) {
    if (callee instanceof Name name) {
      return name.name();
    }
    if (callee instanceof QualifiedExpression qualifiedExpression) {
      return qualifiedExpression.name().name();
    }
    return "Mock";
  }

  private static String patchMessage(List<Argument> arguments, int newPosition) {
    int autospecPosition = newPosition + 4;
    RegularArgument autospecArgument = TreeUtils.nthArgumentOrKeyword(autospecPosition, AUTOSPEC, arguments);
    if (autospecArgument != null && Expressions.isFalsy(autospecArgument.expression())) {
      return AUTOSPEC_FALSE_MESSAGE;
    }
    return PATCH_MESSAGE;
  }

  private static boolean hasReplacement(List<Argument> arguments, int newPosition) {
    return TreeUtils.nthArgumentOrKeyword(newPosition, NEW, arguments) != null
      || TreeUtils.argumentByKeyword(NEW_CALLABLE, arguments) != null;
  }

  private static boolean hasMockSpec(List<Argument> arguments) {
    // Mock(spec, wraps, name, spec_set, ...)
    return TreeUtils.nthArgumentOrKeyword(0, SPEC, arguments) != null
      || TreeUtils.nthArgumentOrKeyword(3, SPEC_SET, arguments) != null;
  }

  private static boolean hasPatchSpecOrAutospec(List<Argument> arguments, int newPosition) {
    // patch(target, new, spec, create, spec_set, autospec, new_callable)
    // patch.object(target, attribute, new, spec, create, spec_set, autospec, new_callable)
    int specPosition = newPosition + 1;
    int specSetPosition = newPosition + 3;
    int autospecPosition = newPosition + 4;
    return TreeUtils.nthArgumentOrKeyword(specPosition, SPEC, arguments) != null
      || TreeUtils.nthArgumentOrKeyword(specSetPosition, SPEC_SET, arguments) != null
      || hasTruthyAutospec(arguments, autospecPosition);
  }

  private static boolean hasTruthyAutospec(List<Argument> arguments, int autospecPosition) {
    RegularArgument autospecArgument = TreeUtils.nthArgumentOrKeyword(autospecPosition, AUTOSPEC, arguments);
    return autospecArgument != null && !Expressions.isFalsy(autospecArgument.expression());
  }
}
