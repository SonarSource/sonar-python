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
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9081")
public class MockPatchLambdaReturnValueCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this lambda with a \"return_value\" argument.";
  private static final String QUICK_FIX_MESSAGE = "Replace this lambda with a \"return_value\" argument";

  private static final String NEW_KEYWORD = "new";
  private static final String PATCH_METHOD = "patch";
  private static final String OBJECT_METHOD = "object";

  /**
   * The {@code mocker} fixtures come from pytest-mock, which has no typeshed stubs, hence the name based detection.
   */
  private static final Set<String> MOCKER_FIXTURE_NAMES = Set.of(
    "mocker", "class_mocker", "module_mocker", "package_mocker", "session_mocker");

  private static final TypeMatcher PATCHER_MATCHER = TypeMatchers.isObjectSatisfying(
    TypeMatchers.any(
      TypeMatchers.isType("unittest.mock._patcher"),
      TypeMatchers.isType("mock.mock._patcher")));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, MockPatchLambdaReturnValueCheck::checkCall);
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

    int newPosition = newArgumentPosition(callExpression.callee(), ctx);
    if (newPosition < 0) {
      return;
    }

    RegularArgument newArgument = TreeUtils.nthArgumentOrKeyword(newPosition, NEW_KEYWORD, arguments);
    if (newArgument == null || !(Expressions.removeParentheses(newArgument.expression()) instanceof LambdaExpression lambda)
      || usesItsParameters(lambda)) {
      return;
    }
    // Decorators evaluate return_value= at definition time; skip lambdas whose body calls something,
    // since that may intentionally re-evaluate on each invocation. Non-decorator patches keep the issue.
    if (TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.DECORATOR) != null
      && containsCallExpression(lambda.expression())) {
      return;
    }

    var issue = ctx.addIssue(lambda, MESSAGE);
    String replacement = returnValueReplacement(callExpression, lambda, newArgument, arguments);
    if (replacement != null) {
      issue.addQuickFix(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
        .addTextEdit(TextEditUtils.replace(newArgument, replacement))
        .build());
    }
  }

  @CheckForNull
  private static String returnValueReplacement(CallExpression callExpression, LambdaExpression lambda, RegularArgument newArgument,
    List<Argument> arguments) {
    if (TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.DECORATOR) != null) {
      // Without "new", the patching call injects a mock into the decorated function, which then needs an extra parameter.
      return null;
    }
    // return_value= evaluates the expression once at patch construction; a call in the lambda body would be re-run per
    // invocation, so rewriting would change behavior for both decorator and with-block forms.
    if (containsCallExpression(lambda.expression())) {
      return null;
    }
    if (newArgument.keywordArgument() == null && hasPositionalArgumentAfter(newArgument, arguments)) {
      // Turning the positional argument into a keyword one would leave positional arguments after a keyword argument.
      return null;
    }
    String returnedValue = TreeUtils.treeToString(lambda.expression(), false);
    if (returnedValue == null) {
      return null;
    }
    return "return_value=" + returnedValue;
  }

  private static boolean hasPositionalArgumentAfter(RegularArgument newArgument, List<Argument> arguments) {
    // Unpacked arguments have already been excluded, hence every argument is a regular one.
    return arguments.stream()
      .dropWhile(argument -> argument != newArgument)
      .skip(1)
      .map(RegularArgument.class::cast)
      .anyMatch(argument -> argument.keywordArgument() == null);
  }

  /**
   * @return the position of the {@code new} parameter for the patching call, or {@code -1} when the callee is not a patching call.
   */
  private static int newArgumentPosition(Expression callee, SubscriptionContext ctx) {
    if (PATCHER_MATCHER.isTrueFor(callee, ctx)) {
      return 1;
    }
    if (!(callee instanceof QualifiedExpression qualifiedExpression)) {
      return -1;
    }
    Expression qualifier = Expressions.removeParentheses(qualifiedExpression.qualifier());
    String memberName = qualifiedExpression.name().name();
    if (OBJECT_METHOD.equals(memberName) && (PATCHER_MATCHER.isTrueFor(qualifier, ctx) || isMockerPatch(qualifier))) {
      return 2;
    }
    if (PATCH_METHOD.equals(memberName) && isMockerFixture(qualifier)) {
      return 1;
    }
    return -1;
  }

  private static boolean isMockerPatch(Expression expression) {
    return expression instanceof QualifiedExpression qualifiedExpression
      && PATCH_METHOD.equals(qualifiedExpression.name().name())
      && isMockerFixture(Expressions.removeParentheses(qualifiedExpression.qualifier()));
  }

  private static boolean isMockerFixture(Expression expression) {
    Expression expr = Expressions.removeParentheses(expression);
    if (expr instanceof Name name) {
      return MOCKER_FIXTURE_NAMES.contains(name.name()) && isFunctionParameter(name.symbolV2());
    }
    // Class-based suites often store the fixture on self in setup_method; match by attribute name only.
    return expr instanceof QualifiedExpression qualifiedExpression
      && Expressions.removeParentheses(qualifiedExpression.qualifier()) instanceof Name qualifier
      && "self".equals(qualifier.name())
      && MOCKER_FIXTURE_NAMES.contains(qualifiedExpression.name().name());
  }

  private static boolean isFunctionParameter(@Nullable SymbolV2 symbol) {
    return Stream.ofNullable(symbol)
      .flatMap(s -> s.usages().stream())
      .anyMatch(usage -> usage.kind() == UsageV2.Kind.PARAMETER);
  }

  private static boolean usesItsParameters(LambdaExpression lambda) {
    ParameterList parameterList = lambda.parameters();
    if (parameterList == null) {
      return false;
    }
    Predicate<Tree> isParameterUsage = tree -> tree instanceof Name name && isDeclaredIn(name.symbolV2(), parameterList);
    Expression body = lambda.expression();
    return isParameterUsage.test(body) || TreeUtils.hasDescendant(body, isParameterUsage);
  }

  private static boolean containsCallExpression(Expression expression) {
    return expression instanceof CallExpression
      || TreeUtils.hasDescendant(expression, CallExpression.class::isInstance);
  }

  /**
   * Names bound by an enclosing scope are also parameters, hence the check that the declaration belongs to the given parameter list.
   */
  private static boolean isDeclaredIn(@Nullable SymbolV2 symbol, ParameterList parameterList) {
    return Stream.ofNullable(symbol)
      .flatMap(s -> s.usages().stream())
      .filter(usage -> usage.kind() == UsageV2.Kind.PARAMETER)
      .anyMatch(usage -> TreeUtils.firstAncestor(usage.tree(), parameterList::equals) != null);
  }
}
