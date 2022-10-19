/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.checks.cdk;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;

import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;
import static org.sonar.python.checks.cdk.CdkPredicate.isString;

public class CdkIamUtils {

  private CdkIamUtils() {
  }

  public static boolean hasNotAllowEffect(@Nullable CdkUtils.ExpressionFlow effect) {
    // default is allow effect
    if (effect == null) {
      return false;
    }
    return !effect.hasExpression(isFqn("aws_cdk.aws_iam.Effect.ALLOW").or(isJsonString("allow")));
  }

  /**
   * In the JSON representation of the PolicyStatement the values are case-insensitive
   */
  private static Predicate<Expression> isJsonString(String expectedValue) {
    return expression -> CdkUtils.getString(expression).filter(expectedValue::equalsIgnoreCase).isPresent();
  }

  /**
   * Examines a list to see if it contains a string that reflects a wildcard and returns this expression as flow.
   */
  private static Optional<CdkUtils.ExpressionFlow> getWildcardInList(SubscriptionContext ctx, ListLiteral list) {
    return CdkUtils.getListElements(ctx, list).stream()
      .filter(expr -> expr.hasExpression(isString("*")))
      .findFirst();
  }

  /**
   * Examines if the flow contains a string that reflects a wildcard and returns this expression as flow.
   */
  public static Optional<CdkUtils.ExpressionFlow> getWildcard(SubscriptionContext ctx, CdkUtils.ExpressionFlow json) {
    if (json.hasExpression(isString("*"))) {
      return Optional.of(json);
    } else {
      return CdkUtils.getList(json).flatMap(list -> CdkIamUtils.getWildcardInList(ctx, list));
    }
  }

  /**
   * Return the json object as dictionary from a form_json call
   */
  public static Optional<DictionaryLiteral> getObjectFromJson(SubscriptionContext ctx, CallExpression call) {
    return CdkUtils.getArgument(ctx, call, "obj", 0).flatMap(CdkUtils::getDictionary);
  }

  /**
   * Return a list of PolicyStatement json representation from a PolicyDocument.from_json call
   */
  public static List<DictionaryLiteral> getPolicyStatements(SubscriptionContext ctx, DictionaryLiteral json) {
    return CdkUtils.getDictionaryPair(ctx, json, "Statement")
      .map(pair -> pair.value)
      .flatMap(CdkUtils::getList)
      .map(list -> CdkUtils.getDictionaryInList(ctx, list))
      .orElse(Collections.emptyList());
  }

  public static CdkUtils.ExpressionFlow getSensitiveExpression(CdkUtils.ExpressionFlow expression, Predicate<Expression> predicate) {
    if (expression.hasExpression(predicate)) {
      return expression;
    } else {
      List<CdkUtils.ExpressionFlow> listElements = CdkUtils.getList(expression)
        .map(list -> CdkUtils.getListElements(expression.ctx(), list))
        .orElse(Collections.emptyList());

      return listElements.stream()
        .filter(expressionFlow -> expressionFlow.hasExpression(predicate))
        .findAny()
        .orElse(null);
    }
  }
}
