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
package org.sonar.python.checks.utils;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

public class UnittestUtils {
  private static final Set<String> PYTEST_LIFECYCLE_METHODS = Set.of(
    "setUp", "tearDown", "setUpClass", "tearDownClass",
    "setup_method", "teardown_method", "setup_class", "teardown_class", "setup_module", "teardown_module");
  private static final String PYTEST_FIXTURE_DECORATOR_FQN = "pytest.fixture";
  private static final String UNITTEST_TEST_CASE_FQN_PREFIX = "unittest.case.TestCase.";
  private static final String PYTEST_EXPECTED_EXCEPTION = "expected_exception";
  private static final String PYTEST_MATCH = "match";
  private static final String UNITTEST_EXCEPTION = "exception";
  private static final TypeMatcher PYTEST_RAISES_MATCHER = TypeMatchers.withFQN("pytest.raises");
  public static final String ASSERT_RAISES_REGEXP = "assertRaisesRegexp";
  public static final String ASSERT_RAISES_REGEX = "assertRaisesRegex";
  private static final TypeMatcher UNITTEST_ASSERT_RAISES_MATCHER = TypeMatchers.any(
    TypeMatchers.isType(UNITTEST_TEST_CASE_FQN_PREFIX + "assertRaises"),
    TypeMatchers.isType(UNITTEST_TEST_CASE_FQN_PREFIX + ASSERT_RAISES_REGEX),
    TypeMatchers.isType(UNITTEST_TEST_CASE_FQN_PREFIX + ASSERT_RAISES_REGEXP)
  );
  private static final TypeMatcher UNITTEST_ASSERT_RAISES_WITH_MESSAGE_CHECK_MATCHER = TypeMatchers.any(
    TypeMatchers.isType(UNITTEST_TEST_CASE_FQN_PREFIX + ASSERT_RAISES_REGEX),
    TypeMatchers.isType(UNITTEST_TEST_CASE_FQN_PREFIX + ASSERT_RAISES_REGEXP)
  );

  private UnittestUtils() {

  }

  public static final Set<String> RUN_METHODS = Set.of("setUp", "tearDown", "setUpClass", "tearDownClass", "run", "skipTest",
    "subTest", "debug", "asyncSetUp", "asyncTearDown");

  public static final Set<String> RAISE_METHODS = Set.of("assertRaises", ASSERT_RAISES_REGEXP, ASSERT_RAISES_REGEX);

  public static final Set<String> ASSERTIONS_METHODS = Set.of("assertEqual", "assertNotEqual", "assertTrue", "assertFalse", "assertIs",
    "assertIsNot", "assertIsNone", "assertIsNotNone", "assertIn", "assertNotIn", "assertIsInstance", "assertNotIsInstance",
    "assertAlmostEqual", "assertNotAlmostEqual", "assertGreater", "assertGreaterEqual", "assertLess", "assertLessEqual",
    "assertRegexpMatches", "assertNotRegexpMatches", "assertItemsEqual", "assertDictContainsSubset", "assertMultiLineEqual",
    "assertSequenceEqual", "assertListEqual", "assertTupleEqual", "assertSetEqual", "assertDictEqual", "assertWarns", "assertWarnsRegex",
    "assertLogs", "assertNoLogs", "assertRegex", "assertNotRegex", "assertCountEqual");

  public static final Set<String> UTIL_METHODS = Set.of("addTypeEqualityFunc", "fail", "failureException", "longMessage", "maxDiff");

  public static final Set<String> GATHER_INFO_METHODS = Set.of("countTestCases", "defaultTestResult", "id", "shortDescription", "addCleanup",
    "doCleanups", "addClassCleanup", "doClassCleanups");

  private static final Set<String> ALL_METHODS = new HashSet<>();
  private static final Set<String> ALL_ASSERT_METHODS = new HashSet<>();

  static {
    ALL_METHODS.addAll(RUN_METHODS);
    ALL_METHODS.addAll(UTIL_METHODS);
    ALL_METHODS.addAll(GATHER_INFO_METHODS);
    ALL_METHODS.addAll(ASSERTIONS_METHODS);
    ALL_METHODS.addAll(RAISE_METHODS);
    ALL_ASSERT_METHODS.addAll(ASSERTIONS_METHODS);
    ALL_ASSERT_METHODS.addAll(RAISE_METHODS);
  }

  public static Set<String> allMethods() {
    return Collections.unmodifiableSet(ALL_METHODS);
  }

  public static Set<String> allAssertMethods() {
    return Collections.unmodifiableSet(ALL_ASSERT_METHODS);
  }

  public static boolean isTestMethodName(String name) {
    return name.startsWith("test");
  }

  public static boolean isTestClassName(String name) {
    return name.startsWith("Test");
  }

  public static boolean isPytestFileName(String fileName) {
    return fileName.startsWith("test_") || fileName.endsWith("_test.py");
  }

  public static boolean hasConstructor(ClassDef classDef) {
    return classDef.body().statements().stream()
      .filter(FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .anyMatch(functionDef -> {
        String name = functionDef.name().name();
        return "__init__".equals(name) || "__new__".equals(name);
      });
  }

  public static boolean hasPytestLifecycleMethods(ClassDef classDef) {
    return classDef.body().statements().stream()
      .filter(FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .anyMatch(functionDef ->
        PYTEST_LIFECYCLE_METHODS.contains(functionDef.name().name())
          || functionDef.decorators().stream()
          .anyMatch(UnittestUtils::isPytestFixtureDecorator));
  }

  public static boolean isPytestStyleTestClass(ClassDef classDef, String fileName) {
    return isPytestFileName(fileName)
      && isTestClassName(classDef.name().name())
      && !hasConstructor(classDef)
      && hasPytestLifecycleMethods(classDef);
  }

  public static boolean isPytestStyleTestFunction(FunctionDef functionDef, String fileName) {
    if (!isPytestFileName(fileName) || !isTestMethodName(functionDef.name().name())) {
      return false;
    }
    ClassDef parentClass = directlyEnclosingClass(functionDef);
    return parentClass == null || isTestClassName(parentClass.name().name());
  }
  public static boolean isPytestRaises(CallExpression callExpression, SubscriptionContext ctx) {
    return PYTEST_RAISES_MATCHER.isTrueFor(callExpression.callee(), ctx);
  }

  public static boolean hasPytestRaisesMatchArgument(CallExpression callExpression) {
    return TreeUtils.argumentByKeyword(PYTEST_MATCH, callExpression.arguments()) != null;
  }

  @Nullable
  public static RegularArgument pytestExpectedExceptionArgument(CallExpression callExpression) {
    return TreeUtils.nthArgumentOrKeyword(0, PYTEST_EXPECTED_EXCEPTION, callExpression.arguments());
  }

  public static boolean isUnittestAssertRaises(CallExpression callExpression, SubscriptionContext ctx) {
    return UNITTEST_ASSERT_RAISES_MATCHER.isTrueFor(callExpression.callee(), ctx);
  }

  public static boolean hasUnittestAssertRaisesMessageCheck(CallExpression callExpression, SubscriptionContext ctx) {
    return UNITTEST_ASSERT_RAISES_WITH_MESSAGE_CHECK_MATCHER.isTrueFor(callExpression.callee(), ctx);
  }

  @Nullable
  public static RegularArgument unittestExceptionArgument(CallExpression callExpression) {
    return TreeUtils.nthArgumentOrKeyword(0, UNITTEST_EXCEPTION, callExpression.arguments());
  }

  private static boolean isPytestFixtureDecorator(Decorator decorator) {
    Expression expression = decorator.expression();
    if (expression instanceof CallExpression callExpression) {
      expression = callExpression.callee();
    }
    return TreeUtils.getSymbolFromTree(expression)
      .map(Symbol::fullyQualifiedName)
      .filter(PYTEST_FIXTURE_DECORATOR_FQN::equals)
      .isPresent();
  }

  public static boolean isUnittestTestCaseClass(ClassDef classDef) {
    return TreeUtils.getParentClassesFQN(classDef).stream().anyMatch(parentClass -> parentClass.contains("unittest") && parentClass.contains("TestCase"));
  }

  public static boolean isWithinUnittestTestCase(Tree tree) {
    ClassDef classDef = (ClassDef) TreeUtils.firstAncestorOfKind(tree, Tree.Kind.CLASSDEF);
    return classDef != null && isUnittestTestCaseClass(classDef);
  }

  @Nullable
  private static ClassDef directlyEnclosingClass(Tree tree) {
    Tree current = tree.parent();
    while (current != null) {
      if (current.is(Tree.Kind.CLASSDEF)) {
        return (ClassDef) current;
      }
      if (current.is(Tree.Kind.FUNCDEF, Tree.Kind.LAMBDA)) {
        return null;
      }
      current = current.parent();
    }
    return null;
  }
}
