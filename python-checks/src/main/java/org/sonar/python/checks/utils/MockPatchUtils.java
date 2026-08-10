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

import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

public final class MockPatchUtils {

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

  private MockPatchUtils() {
  }

  /**
   * @return the position of the {@code new} parameter for the patching call, or {@code -1} when the callee is not a patching call.
   */
  public static int newArgumentPosition(Expression callee, SubscriptionContext ctx) {
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
}
