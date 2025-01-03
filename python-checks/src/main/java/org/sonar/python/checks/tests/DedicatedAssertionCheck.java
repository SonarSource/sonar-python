/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.checks.tests;

import java.util.List;
import java.util.Map;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.InExpression;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tests.UnittestUtils;
import org.sonar.python.tree.TreeUtils;


@Rule(key = "S5906")
public class DedicatedAssertionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Consider using \"%s\" instead.";
  private static final String MESSAGE_ROUND_ALMOST_EQUAL = "Consider using the \"places\" argument of \"%s\" instead.";

  private static final String ASSERT_TRUE = "assertTrue";
  private static final String ASSERT_FALSE = "assertFalse";
  private static final String ASSERT_EQUAL = "assertEqual";
  private static final String ASSERT_NOT_EQUAL = "assertNotEqual";
  private static final String ASSERT_ALMOST_EQUAL = "assertAlmostEqual";
  private static final String ASSERT_NOT_ALMOST_EQUAL = "assertNotAlmostEqual";

  private static final String IS = "#is#";
  private static final String IS_NOT = "#is-not#";
  private static final String IN = "#in#";
  private static final String NOT_IN = "#not-in#";

  private static final String FIRST = "first";
  private static final String SECOND = "second";

  private static final String ROUND = "round";
  private static final String IS_INSTANCE = "isinstance";

  private static final Map<String, String> ASSERT_TRUE_SUGGESTIONS = Map.ofEntries(
    Map.entry("==", ASSERT_EQUAL),
    Map.entry("!=", ASSERT_NOT_EQUAL),
    Map.entry(">", "assertGreater"),
    Map.entry(">=", "assertGreaterEqual"),
    Map.entry("<", "assertLess"),
    Map.entry("<=", "assertLessEqual"),
    Map.entry(IN, "assertIn"),
    Map.entry(NOT_IN, "assertNotIn"),
    Map.entry(IS, "assertIs"),
    Map.entry(IS_NOT, "assertIsNot"),
    Map.entry(IS_INSTANCE, "assertIsInstance")
  );

  private static final Map<String, String> ASSERT_FALSE_SUGGESTIONS = Map.of(
    "==", ASSERT_NOT_EQUAL,
    "!=", ASSERT_EQUAL,
    IN, "assertNotIn",
    NOT_IN, "assertIn",
    IS, "assertIsNot",
    IS_NOT, "assertIs",
    IS_INSTANCE, "assertNotIsInstance"
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      if (!UnittestUtils.isWithinUnittestTestCase(callExpression)) {
        return;
      }
      Expression callee = callExpression.callee();
      if (!callee.is(Tree.Kind.QUALIFIED_EXPR)) {
        return;
      }
      QualifiedExpression qualifiedExpression = (QualifiedExpression) callee;
      if (!qualifiedExpression.qualifier().is(Tree.Kind.NAME) || !((Name) qualifiedExpression.qualifier()).name().equals("self")) {
        return;
      }
      String calledMethod = qualifiedExpression.name().name();

      switch (calledMethod) {
        case ASSERT_TRUE:
          checkAssertTrueOrFalse(callExpression, ctx, true);
          break;
        case ASSERT_FALSE:
          checkAssertTrueOrFalse(callExpression, ctx, false);
          break;
        case ASSERT_EQUAL:
          checkAssertEqualOrNotEqual(callExpression, ctx, true);
          break;
        case ASSERT_NOT_EQUAL:
          checkAssertEqualOrNotEqual(callExpression, ctx, false);
          break;
        case ASSERT_ALMOST_EQUAL:
          checkAssertAlmostEqualOrNot(callExpression, ctx, ASSERT_ALMOST_EQUAL);
          break;
        case ASSERT_NOT_ALMOST_EQUAL:
          checkAssertAlmostEqualOrNot(callExpression, ctx, ASSERT_NOT_ALMOST_EQUAL);
          break;
        default:
          // Nothing to do
      }
    });
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }
  
  private static void checkAssertTrueOrFalse(CallExpression callExpression, SubscriptionContext ctx, boolean isAssertTrue) {
    Map<String, String> suggestions = isAssertTrue ? ASSERT_TRUE_SUGGESTIONS : ASSERT_FALSE_SUGGESTIONS;
    List<Argument> arguments = callExpression.arguments();
    if (arguments.isEmpty()) {
      return;
    }
    RegularArgument firstArg = TreeUtils.nthArgumentOrKeyword(0, "expr", arguments);
    if (firstArg == null) {
      return;
    }
    Expression expression = Expressions.removeParentheses(firstArg.expression());
    String dedicatedAssertion = null;
    if (expression.is(Tree.Kind.COMPARISON)) {
      BinaryExpression binaryExpression = (BinaryExpression) expression;
      if (binaryExpression.leftOperand().is(Tree.Kind.COMPARISON)) {
        // Avoid reporting on chained comparisons (due to how comparisons are defined in the grammar, only the left operand can be a comparison)
        return;
      }
      dedicatedAssertion = suggestions.get(binaryExpression.operator().value());
    } else if (expression.is(Tree.Kind.IN)) {
      dedicatedAssertion = ((InExpression) expression).notToken() == null ? suggestions.get(IN) : suggestions.get(NOT_IN);
    } else if (expression.is(Tree.Kind.IS)) {
      dedicatedAssertion = ((IsExpression) expression).notToken() == null ? suggestions.get(IS) : suggestions.get(IS_NOT);
    } else if (isCallTo(IS_INSTANCE, expression)) {
      dedicatedAssertion = suggestions.get(IS_INSTANCE);
    }
    if (dedicatedAssertion != null) {
      ctx.addIssue(callExpression, String.format(MESSAGE, dedicatedAssertion));
    }
  }

  private static void checkAssertEqualOrNotEqual(CallExpression callExpression, SubscriptionContext ctx, boolean isAssertEqual) {
    List<Argument> arguments = callExpression.arguments();
    RegularArgument firstArg = TreeUtils.nthArgumentOrKeyword(0, FIRST, arguments);
    RegularArgument secondArg = TreeUtils.nthArgumentOrKeyword(1, SECOND, arguments);
    if (firstArg == null || secondArg == null) {
      return;
    }

    Expression firstExpression = Expressions.removeParentheses(firstArg.expression());
    Expression secondExpression = Expressions.removeParentheses(secondArg.expression());
    String firstDedicatedAssertion = dedicatedAssertion(firstExpression, isAssertEqual);
    if (firstDedicatedAssertion != null) {
      // If two suggestions are possible, priority is given to first argument, as it is conventionally the "expected" value
      ctx.addIssue(callExpression, String.format(MESSAGE, firstDedicatedAssertion));
      return;
    }
    String secondDedicatedAssertion = dedicatedAssertion(secondExpression, isAssertEqual);
    if (secondDedicatedAssertion != null) {
      ctx.addIssue(callExpression, String.format(MESSAGE, secondDedicatedAssertion));
    }
  }

  private static void checkAssertAlmostEqualOrNot(CallExpression callExpression, SubscriptionContext ctx, String methodName) {
    List<Argument> arguments = callExpression.arguments();
    RegularArgument firstArg = TreeUtils.nthArgumentOrKeyword(0, FIRST, arguments);
    RegularArgument secondArg = TreeUtils.nthArgumentOrKeyword(1, SECOND, arguments);
    if (firstArg == null || secondArg == null) {
      return;
    }

    Expression firstExpression = Expressions.removeParentheses(firstArg.expression());
    Expression secondExpression = Expressions.removeParentheses(secondArg.expression());
    if (isCallTo(ROUND, firstExpression) || isCallTo(ROUND, secondExpression)) {
      ctx.addIssue(callExpression, String.format(MESSAGE_ROUND_ALMOST_EQUAL, methodName));
    }
  }

  @CheckForNull
  private static String dedicatedAssertion(Expression expression, boolean isAssertEqual) {
    if (expression.is(Tree.Kind.NAME)) {
      String name = ((Name) expression).name();
      if (name.equals("True")) {
        return isAssertEqual ? ASSERT_TRUE : null;
      }
      if (name.equals("False")) {
        return isAssertEqual ? ASSERT_FALSE : null;
      }
    }
    if (expression.is(Tree.Kind.NONE)) {
      return isAssertEqual ? "assertIsNone" : "assertIsNotNone";
    }
    if (isCallTo(ROUND, expression)) {
      return isAssertEqual ? ASSERT_ALMOST_EQUAL : ASSERT_NOT_ALMOST_EQUAL;
    }
    return null;
  }

  private static boolean isCallTo(String functionName, Expression expression) {
    if (expression.is(Tree.Kind.CALL_EXPR)) {
      CallExpression assertedCall = (CallExpression) expression;
      Expression callee = assertedCall.callee();
      if (!callee.is(Tree.Kind.NAME)) {
        return false;
      }
      return functionName.equals(((Name) callee).name());
    }
    return false;
  }
}
