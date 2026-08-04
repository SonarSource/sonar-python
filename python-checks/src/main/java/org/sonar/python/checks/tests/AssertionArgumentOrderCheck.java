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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TokenLocation;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.UnittestUtils;
import org.sonar.python.checks.utils.UnittestUtils.AssertionArguments;
import org.sonar.python.checks.utils.UnittestUtils.AssertionFrameworkHandlers;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.utils.UnittestUtils.ASSERTPY_IS_EQUAL_TO_MATCHER;
import static org.sonar.python.checks.utils.UnittestUtils.PYTEST_APPROX_MATCHER;

@Rule(key = "S3415")
public class AssertionArgumentOrderCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Unify assertion argument order in this file; both \"actual first\" and \"expected first\" conventions are used.";
  private static final String ACTUAL_FIRST_FLOW = "Actual value first";
  private static final String EXPECTED_FIRST_FLOW = "Expected value first";
  private static final String ACTUAL_FIRST_LOCATION_MESSAGE = "Actual value first.";
  private static final String EXPECTED_FIRST_LOCATION_MESSAGE = "Expected value first.";
  private static final String PUT_EXPECTED_SECOND_QF = "Put all expected values second";
  private static final String PUT_ACTUAL_SECOND_QF = "Put all actual values second";
  private static final String PYTEST_APPROX_EXPECTED_ARGUMENT_NAME = "expected";

  private enum Convention {
    ACTUAL_FIRST,
    EXPECTED_FIRST
  }

  private final List<OrderableAssertion> assertions = new ArrayList<>();
  private SubscriptionContext subscriptionContext;

  private record OrderableAssertion(
    Tree primaryLocation,
    Convention convention,
    List<PythonTextEdit> swapEdits) {
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      assertions.clear();
      subscriptionContext = ctx;
    });
    UnittestUtils.registerAssertionSyntaxNodeConsumers(context, new AssertionFrameworkHandlers(
      this::checkUnittestAssertion,
      this::checkAssertpyAssertion,
      this::checkPytestAssertion));
  }

  @Override
  public void leaveFile() {
    if (subscriptionContext == null) {
      return;
    }

    List<OrderableAssertion> actualFirst = assertions.stream().filter(a -> a.convention() == Convention.ACTUAL_FIRST).toList();
    List<OrderableAssertion> expectedFirst = assertions.stream().filter(a -> a.convention() == Convention.EXPECTED_FIRST).toList();
    if (actualFirst.isEmpty() || expectedFirst.isEmpty()) {
      return;
    }

    OrderableAssertion primary = assertions.get(0);
    PythonCheck.PreciseIssue issue = subscriptionContext.addIssue(primary.primaryLocation(), MESSAGE);
    issue.addFlow(ACTUAL_FIRST_FLOW, toFlowLocations(actualFirst, ACTUAL_FIRST_LOCATION_MESSAGE));
    issue.addFlow(EXPECTED_FIRST_FLOW, toFlowLocations(expectedFirst, EXPECTED_FIRST_LOCATION_MESSAGE));
    createUnifyQuickFix(PUT_EXPECTED_SECOND_QF, expectedFirst).ifPresent(issue::addQuickFix);
    createUnifyQuickFix(PUT_ACTUAL_SECOND_QF, actualFirst).ifPresent(issue::addQuickFix);
  }

  @Override
  public CheckScope scope() {
    return CheckScope.TESTS;
  }

  private void checkUnittestAssertion(SubscriptionContext ctx, CallExpression callExpression) {
    AssertionArguments arguments = UnittestUtils.unittestAssertionArguments(callExpression, ctx);
    if (arguments == null) {
      return;
    }
    classifyAndCollect(callExpression, arguments.actual(), arguments.expected(), ctx);
  }

  private void checkPytestAssertion(SubscriptionContext ctx, AssertStatement assertStatement) {
    if (!UnittestUtils.isPytestStyleTestFunction(ctx, assertStatement)) {
      return;
    }

    Expression condition = Expressions.removeParentheses(assertStatement.condition());
    if (!(condition instanceof BinaryExpression binaryExpression) || !"==".equals(binaryExpression.operator().value())) {
      return;
    }

    Expression left = binaryExpression.leftOperand();
    Expression right = binaryExpression.rightOperand();
    boolean leftExpected = isExpectedValue(left, ctx);
    boolean rightExpected = isExpectedValue(right, ctx);
    if (leftExpected == rightExpected) {
      return;
    }

    Convention convention = leftExpected ? Convention.EXPECTED_FIRST : Convention.ACTUAL_FIRST;
    List<PythonTextEdit> swapEdits = createPytestSwapEdits(binaryExpression, convention, ctx);
    assertions.add(new OrderableAssertion(condition, convention, swapEdits));
  }

  private void checkAssertpyAssertion(SubscriptionContext ctx, CallExpression callExpression) {
    AssertionArguments arguments = UnittestUtils.assertpyAssertionArguments(callExpression, ctx, ASSERTPY_IS_EQUAL_TO_MATCHER);
    if (arguments == null) {
      return;
    }
    classifyAndCollect(callExpression, arguments.actual(), arguments.expected(), ctx);
  }

  private void classifyAndCollect(Tree primaryLocation, Expression firstPosition, Expression secondPosition, SubscriptionContext ctx) {
    boolean firstExpected = isExpectedValue(firstPosition, ctx);
    boolean secondExpected = isExpectedValue(secondPosition, ctx);
    if (firstExpected == secondExpected) {
      return;
    }
    Convention convention = firstExpected ? Convention.EXPECTED_FIRST : Convention.ACTUAL_FIRST;
    assertions.add(new OrderableAssertion(primaryLocation, convention, createSwapEdits(firstPosition, secondPosition, ctx)));
  }

  private static List<IssueLocation> toFlowLocations(List<OrderableAssertion> group, String message) {
    return group.stream()
      .map(assertion -> IssueLocation.preciseLocation(assertion.primaryLocation(), message))
      .toList();
  }

  private static Optional<PythonQuickFix> createUnifyQuickFix(String description, List<OrderableAssertion> toSwap) {
    // Only offer a unify fix when every assertion in the group can be edited; otherwise applying a
    // partial fix would leave the file still mixing conventions.
    if (toSwap.stream().anyMatch(assertion -> assertion.swapEdits().isEmpty())) {
      return Optional.empty();
    }
    List<PythonTextEdit> edits = toSwap.stream().flatMap(assertion -> assertion.swapEdits().stream()).toList();
    return Optional.of(PythonQuickFix.newQuickFix(description).addTextEdit(edits).build());
  }

  private static List<PythonTextEdit> createSwapEdits(Expression leftExpression, Expression rightExpression, SubscriptionContext ctx) {
    String leftText = expressionText(leftExpression, ctx);
    String rightText = expressionText(rightExpression, ctx);
    if (leftText == null || rightText == null) {
      return List.of();
    }
    return List.of(
      TextEditUtils.replace(leftExpression, rightText),
      TextEditUtils.replace(rightExpression, leftText));
  }

  private static List<PythonTextEdit> createPytestSwapEdits(BinaryExpression binaryExpression, Convention convention, SubscriptionContext ctx) {
    Expression left = binaryExpression.leftOperand();
    Expression right = binaryExpression.rightOperand();
    Expression expectedOperand = convention == Convention.EXPECTED_FIRST ? left : right;
    Expression actualOperand = convention == Convention.EXPECTED_FIRST ? right : left;

    // expected == approx(actual) (or mirrored): unwrap by swapping the constant with the approx argument
    CallExpression approxWrappingActual = asPytestApproxCall(actualOperand, ctx);
    if (approxWrappingActual != null) {
      RegularArgument approxExpectedArg = TreeUtils.nthArgumentOrKeyword(0, PYTEST_APPROX_EXPECTED_ARGUMENT_NAME, approxWrappingActual.arguments());
      if (approxExpectedArg == null) {
        return List.of();
      }
      String replacementForExpectedOperand = expressionText(approxExpectedArg.expression(), ctx);
      String replacementForApproxArg = expressionText(expectedOperand, ctx);
      if (replacementForExpectedOperand == null || replacementForApproxArg == null) {
        return List.of();
      }
      return List.of(
        TextEditUtils.replace(expectedOperand, replacementForExpectedOperand),
        TextEditUtils.replace(approxExpectedArg.expression(), replacementForApproxArg));
    }

    return createSwapEdits(left, right, ctx);
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
}
