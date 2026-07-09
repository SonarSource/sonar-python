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
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TokenLocation;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.AssertpyUtils;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.UnittestUtils;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S3415")
public class AssertionArgumentOrderCheck extends PythonSubscriptionCheck {
  private static final boolean DEFAULT_EXPECTED_ON_RIGHT = true;
  private static final String UNITTEST_MESSAGE = "Swap these 2 arguments so they are in the correct order: actual value, expected value.";
  private static final String PYTEST_EXPECTED_ON_RIGHT_MESSAGE = "Swap these 2 sides so they are in the correct order: actual value, expected value.";
  private static final String PYTEST_EXPECTED_ON_LEFT_MESSAGE = "Swap these 2 sides so they are in the correct order: expected value, actual value.";
  private static final String ASSERTPY_MESSAGE = "Pass the actual value to \"assert_that\" and the expected value to \"is_equal_to\".";
  private static final String EXPECTED_SECONDARY_MESSAGE = "Expected value.";
  private static final String ACTUAL_SECONDARY_MESSAGE = "Actual value.";
  private static final String UNITTEST_QUICK_FIX_MESSAGE = "Swap the actual and expected arguments";
  private static final String PYTEST_QUICK_FIX_MESSAGE = "Swap the actual and expected operands";
  private static final String ASSERTPY_QUICK_FIX_MESSAGE = "Swap the actual and expected values";
  private static final String PYTEST_APPROX_EXPECTED_ARGUMENT_NAME = "expected";

  private static final TypeMatcher UNITTEST_EQUALITY_ASSERTION_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("unittest.case.TestCase.assertEqual"),
    TypeMatchers.isType("unittest.case.TestCase.assertNotEqual"),
    TypeMatchers.isType("unittest.case.TestCase.assertAlmostEqual"),
    TypeMatchers.isType("unittest.case.TestCase.assertNotAlmostEqual"));
  private static final TypeMatcher UNITTEST_IDENTITY_ASSERTION_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("unittest.case.TestCase.assertIs"),
    TypeMatchers.isType("unittest.case.TestCase.assertIsNot"));
  private static final TypeMatcher PYTEST_APPROX_MATCHER = TypeMatchers.isType("pytest.approx");
  private static final TypeMatcher ASSERTPY_IS_EQUAL_TO_MATCHER = TypeMatchers.isType("assertpy.AssertionBuilder.is_equal_to");

  @RuleProperty(
    key = "expectedOnRight",
    description = "Whether the expected value should be on the right-hand side of pytest equality assertions.",
    defaultValue = "" + DEFAULT_EXPECTED_ON_RIGHT)
  public boolean expectedOnRight = DEFAULT_EXPECTED_ON_RIGHT;

  private record ParameterNames(String leftKeyword, String rightKeyword) {
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      checkUnittestAssertion(ctx, callExpression);
      checkAssertpyAssertion(ctx, callExpression);
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSERT_STMT, ctx -> checkPytestAssertion(ctx, (AssertStatement) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.TESTS;
  }

  private static void checkUnittestAssertion(SubscriptionContext ctx, CallExpression callExpression) {
    if (!UnittestUtils.isWithinUnittestTestCase(callExpression)) {
      return;
    }
    if (!(Expressions.removeParentheses(callExpression.callee()) instanceof QualifiedExpression qualifiedExpression)
      || !CheckUtils.isSelf(qualifiedExpression.qualifier())) {
      return;
    }

    ParameterNames parameterNames = parameterNames(callExpression.callee(), ctx);
    if (parameterNames == null) {
      return;
    }

    List<Argument> arguments = callExpression.arguments();
    RegularArgument firstArg = TreeUtils.nthArgumentOrKeyword(0, parameterNames.leftKeyword(), arguments);
    RegularArgument secondArg = TreeUtils.nthArgumentOrKeyword(1, parameterNames.rightKeyword(), arguments);
    if (firstArg == null || secondArg == null) {
      return;
    }

    if (areInvertedForActualExpected(firstArg.expression(), secondArg.expression(), ctx)) {
      var issue = ctx.addIssue(callExpression, UNITTEST_MESSAGE);
      addActualExpectedSecondaryLocations(issue, secondArg.expression(), firstArg.expression());
      createSwapQuickFix(firstArg.expression(), secondArg.expression(), UNITTEST_QUICK_FIX_MESSAGE, ctx).ifPresent(issue::addQuickFix);
    }
  }

  private void checkPytestAssertion(SubscriptionContext ctx, AssertStatement assertStatement) {
    if (!isPytestStyleTestFunction(ctx, assertStatement)) {
      return;
    }

    Expression condition = Expressions.removeParentheses(assertStatement.condition());
    if (!(condition instanceof BinaryExpression binaryExpression) || !"==".equals(binaryExpression.operator().value())) {
      return;
    }

    if (areInvertedForConfiguredPytestOrder(binaryExpression.leftOperand(), binaryExpression.rightOperand(), ctx)) {
      var issue = ctx.addIssue(condition, pytestMessage());
      addActualExpectedSecondaryLocations(issue, pytestActualExpression(binaryExpression, ctx), pytestExpectedExpression(binaryExpression, ctx));
      createPytestQuickFix(binaryExpression, ctx).ifPresent(issue::addQuickFix);
    }
  }

  private static void checkAssertpyAssertion(SubscriptionContext ctx, CallExpression callExpression) {
    if (!isSupportedTestFunction(ctx, callExpression)) {
      return;
    }
    if (!(Expressions.removeParentheses(callExpression.callee()) instanceof QualifiedExpression qualifiedExpression)
      || !ASSERTPY_IS_EQUAL_TO_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }

    CallExpression assertThatCall = AssertpyUtils.originatingAssertThatCall(qualifiedExpression.qualifier(), ctx);
    if (assertThatCall == null) {
      return;
    }

    RegularArgument actualArg = TreeUtils.nthArgumentOrKeyword(0, "val", assertThatCall.arguments());
    RegularArgument expectedArg = TreeUtils.nthArgumentOrKeyword(0, "other", callExpression.arguments());
    if (actualArg == null || expectedArg == null) {
      return;
    }

    if (areInvertedForActualExpected(actualArg.expression(), expectedArg.expression(), ctx)) {
      var issue = ctx.addIssue(callExpression, ASSERTPY_MESSAGE);
      // On the noncompliant path the assertion arguments are inverted: the assert_that slot
      // holds the expected value and the is_equal_to slot holds the actual value.
      addActualExpectedSecondaryLocations(issue, expectedArg.expression(), actualArg.expression());
      createSwapQuickFix(actualArg.expression(), expectedArg.expression(), ASSERTPY_QUICK_FIX_MESSAGE, ctx).ifPresent(issue::addQuickFix);
    }
  }

  private static void addActualExpectedSecondaryLocations(org.sonar.plugins.python.api.PythonCheck.PreciseIssue issue, Expression actualExpression, Expression expectedExpression) {
    issue.secondary(expectedExpression, EXPECTED_SECONDARY_MESSAGE);
    issue.secondary(actualExpression, ACTUAL_SECONDARY_MESSAGE);
  }

  private Optional<PythonQuickFix> createPytestQuickFix(BinaryExpression binaryExpression, SubscriptionContext ctx) {
    Expression actualOperand = expectedOnRight ? binaryExpression.leftOperand() : binaryExpression.rightOperand();
    Expression expectedOperand = expectedOnRight ? binaryExpression.rightOperand() : binaryExpression.leftOperand();
    CallExpression approxCall = asPytestApproxCall(expectedOperand, ctx);
    if (approxCall == null) {
      return createSwapQuickFix(binaryExpression.leftOperand(), binaryExpression.rightOperand(), PYTEST_QUICK_FIX_MESSAGE, ctx);
    }

    RegularArgument approxExpectedArg = TreeUtils.nthArgumentOrKeyword(0, PYTEST_APPROX_EXPECTED_ARGUMENT_NAME, approxCall.arguments());
    if (approxExpectedArg == null) {
      return Optional.empty();
    }

    String replacementForOperand = expressionText(approxExpectedArg.expression(), ctx);
    String replacementForApproxExpectedArg = expressionText(actualOperand, ctx);
    if (replacementForOperand == null || replacementForApproxExpectedArg == null) {
      return Optional.empty();
    }

    return Optional.of(PythonQuickFix.newQuickFix(PYTEST_QUICK_FIX_MESSAGE)
      .addTextEdit(TextEditUtils.replace(actualOperand, replacementForOperand))
      .addTextEdit(TextEditUtils.replace(approxExpectedArg.expression(), replacementForApproxExpectedArg))
      .build());
  }

  private static Optional<PythonQuickFix> createSwapQuickFix(Expression leftExpression, Expression rightExpression, String message, SubscriptionContext ctx) {
    String leftText = expressionText(leftExpression, ctx);
    String rightText = expressionText(rightExpression, ctx);
    if (leftText == null || rightText == null) {
      return Optional.empty();
    }
    return Optional.of(PythonQuickFix.newQuickFix(message)
      .addTextEdit(TextEditUtils.replace(leftExpression, rightText))
      .addTextEdit(TextEditUtils.replace(rightExpression, leftText))
      .build());
  }

  @Nullable
  private static CallExpression asPytestApproxCall(Expression expression, SubscriptionContext ctx) {
    Expression normalized = Expressions.removeParentheses(expression);
    if (normalized instanceof CallExpression callExpression && PYTEST_APPROX_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      return callExpression;
    }
    return null;
  }

  @Nullable
  private static String expressionText(Expression expression, SubscriptionContext ctx) {
    String fileContent = ctx.pythonFile().content();
    int startIndex = convertPositionToIndex(fileContent, expression.firstToken().line(), expression.firstToken().column());
    TokenLocation endLocation = new TokenLocation(expression.lastToken());
    int endIndex = convertPositionToIndex(fileContent, endLocation.endLine(), endLocation.endLineOffset());
    if (startIndex < 0 || endIndex < startIndex || endIndex > fileContent.length()) {
      return null;
    }
    return fileContent.substring(startIndex, endIndex);
  }

  private static int convertPositionToIndex(String fileContent, int line, int lineOffset) {
    int currentIndex = 0;
    int currentLine = 1;

    while (currentLine < line && currentIndex < fileContent.length()) {
      int nextIndex = nextIndex(fileContent, currentIndex);
      if (isLineBreak(fileContent.charAt(currentIndex))) {
        currentLine++;
      }
      currentIndex = nextIndex;
    }

    if (currentLine != line) {
      return -1;
    }

    int lineEnd = findLineEnd(fileContent, currentIndex);

    if (lineOffset < 0 || currentIndex + lineOffset > lineEnd) {
      return -1;
    }
    return currentIndex + lineOffset;
  }

  private static int nextIndex(String fileContent, int currentIndex) {
    char current = fileContent.charAt(currentIndex);
    currentIndex++;
    if (current == '\r') {
      if (currentIndex < fileContent.length() && fileContent.charAt(currentIndex) == '\n') {
        return currentIndex + 1;
      }
      return currentIndex;
    }
    if (current == '\n') {
      return currentIndex;
    }
    return currentIndex;
  }

  private static int findLineEnd(String fileContent, int currentIndex) {
    int lineEnd = currentIndex;
    while (lineEnd < fileContent.length() && !isLineBreak(fileContent.charAt(lineEnd))) {
      lineEnd++;
    }
    return lineEnd;
  }

  private static boolean isLineBreak(char current) {
    return current == '\r' || current == '\n';
  }

  @Nullable
  private static ParameterNames parameterNames(Expression callee, SubscriptionContext ctx) {
    if (UNITTEST_EQUALITY_ASSERTION_MATCHER.isTrueFor(callee, ctx)) {
      return new ParameterNames("first", "second");
    }
    if (UNITTEST_IDENTITY_ASSERTION_MATCHER.isTrueFor(callee, ctx)) {
      return new ParameterNames("expr1", "expr2");
    }
    return null;
  }

  private static boolean areInvertedForActualExpected(Expression actualPosition, Expression expectedPosition, SubscriptionContext ctx) {
    return isExpectedValue(actualPosition, ctx) && !isExpectedValue(expectedPosition, ctx);
  }

  private boolean areInvertedForConfiguredPytestOrder(Expression leftOperand, Expression rightOperand, SubscriptionContext ctx) {
    return expectedOnRight
      ? areInvertedForActualExpected(leftOperand, rightOperand, ctx)
      : areInvertedForActualExpected(rightOperand, leftOperand, ctx);
  }

  private String pytestMessage() {
    return expectedOnRight ? PYTEST_EXPECTED_ON_RIGHT_MESSAGE : PYTEST_EXPECTED_ON_LEFT_MESSAGE;
  }

  private Expression pytestActualExpression(BinaryExpression binaryExpression, SubscriptionContext ctx) {
    Expression actualExpression = expectedOnRight ? binaryExpression.rightOperand() : binaryExpression.leftOperand();
    return unwrapPytestApproxArgument(actualExpression, ctx);
  }

  private Expression pytestExpectedExpression(BinaryExpression binaryExpression, SubscriptionContext ctx) {
    Expression expectedExpression = expectedOnRight ? binaryExpression.leftOperand() : binaryExpression.rightOperand();
    return unwrapPytestApproxArgument(expectedExpression, ctx);
  }

  private static boolean isExpectedValue(Expression expression, SubscriptionContext ctx) {
    Expression unwrapped = Expressions.removeParentheses(expression);
    if (CheckUtils.isConstant(unwrapped)) {
      return true;
    }
    if (isPytestApproxExpectedValue(unwrapped, ctx)) {
      return true;
    }
    if (unwrapped instanceof Name name) {
      return Expressions.singleAssignedNonNameValue(name).filter(CheckUtils::isImmutableConstant).isPresent();
    }
    return false;
  }

  private static boolean isPytestApproxExpectedValue(Expression expression, SubscriptionContext ctx) {
    if (!(expression instanceof CallExpression callExpression) || !PYTEST_APPROX_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      return false;
    }
    RegularArgument expectedArg = TreeUtils.nthArgumentOrKeyword(0, PYTEST_APPROX_EXPECTED_ARGUMENT_NAME, callExpression.arguments());
    return expectedArg != null && isExpectedValue(expectedArg.expression(), ctx);
  }

  private static Expression unwrapPytestApproxArgument(Expression expression, SubscriptionContext ctx) {
    CallExpression approxCall = asPytestApproxCall(expression, ctx);
    if (approxCall == null) {
      return expression;
    }
    RegularArgument expectedArg = TreeUtils.nthArgumentOrKeyword(0, PYTEST_APPROX_EXPECTED_ARGUMENT_NAME, approxCall.arguments());
    return expectedArg != null ? expectedArg.expression() : expression;
  }

  private static boolean isSupportedTestFunction(SubscriptionContext ctx, Tree tree) {
    FunctionDef functionDef = (FunctionDef) TreeUtils.firstAncestorOfKind(tree, Tree.Kind.FUNCDEF);
    return functionDef != null && (UnittestUtils.isWithinUnittestTestCase(functionDef) || UnittestUtils.isPytestStyleTestFunction(functionDef, ctx.pythonFile().fileName()));
  }

  private static boolean isPytestStyleTestFunction(SubscriptionContext ctx, Tree tree) {
    FunctionDef functionDef = (FunctionDef) TreeUtils.firstAncestorOfKind(tree, Tree.Kind.FUNCDEF);
    return functionDef != null && UnittestUtils.isPytestStyleTestFunction(functionDef, ctx.pythonFile().fileName());
  }
}
