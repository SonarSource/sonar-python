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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.StringLiteral;

import static org.sonar.python.checks.cdk.CdkPredicate.isNumericLiteral;
import static org.sonar.python.checks.cdk.CdkPredicate.isString;
import static org.sonar.python.checks.cdk.CdkPredicate.isStringLiteral;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;
import static org.sonar.python.checks.cdk.CdkUtils.getCall;
import static org.sonar.python.checks.cdk.CdkUtils.getDictionary;

@Rule(key = "S6321")
public class UnrestrictedAdministrationCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    new UnrestrictedAdministrationCheckPartCfnSecurity().initialize(context);
    new UnrestrictedAdministrationCheckPartConnections().initialize(context);
  }

  // Utils elements used by each classes implementing the rule
  static class Call {
    SubscriptionContext ctx;
    CallExpression callExpression;

    public Call(SubscriptionContext ctx, CallExpression callExpression) {
      this.ctx = ctx;
      this.callExpression = callExpression;
    }

    boolean hasArgument(String name, Predicate<Expression> predicate) {
      return hasArgument(name, -1, predicate);
    }

    boolean hasArgument(String name, int pos, Predicate<Expression> predicate) {
      return getArgument(ctx, callExpression, name, pos).filter(flow -> flow.hasExpression(predicate)).isPresent();
    }

    public Optional<Long> getArgumentAsLong(String name) {
      return getArgument(ctx, callExpression, name)
        .flatMap(flow -> flow.getExpression(isNumericLiteral()))
        .map(NumericLiteral.class::cast)
        .map(NumericLiteral::valueAsLong);
    }

    boolean hasSensitivePortRange(String minName, String maxName, long[] numbers) {
      Optional<Long> min = getArgumentAsLong(minName);
      Optional<Long> max = getArgumentAsLong(maxName);

      if (min.isEmpty() || max.isEmpty()) {
        return false;
      }

      return isInInterval(min.get(), max.get(), numbers);
    }
  }

  public static boolean isInInterval(long min, long max, long[] numbers) {
    for (long port : numbers) {
      if (min <= port && port <= max) {
        return true;
      }
    }
    return false;
  }
}
