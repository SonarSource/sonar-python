/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.checks;

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6740")
public class PandasReadNoDataTypeCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "The 'dtype' parameter should be used in calls to";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, PandasReadNoDataTypeCheck::checkReadMethodCall);
  }

  private static void checkReadMethodCall(SubscriptionContext subscriptionContext) {
    CallExpression callExpression = (CallExpression) subscriptionContext.syntaxNode();
    Optional.of(callExpression)
      .filter(PandasReadNoDataTypeCheck::isReadCall)
      .filter(ce -> TreeUtils.nthArgumentOrKeyword(1, "dtype", ce.arguments()) == null)
      .map(PandasReadNoDataTypeCheck::getMessage)
      .ifPresent(message -> subscriptionContext.addIssue(callExpression.callee().lastToken(), message));
  }

  private static boolean isReadCall(CallExpression callExpression) {
    return Optional.of(callExpression)
      .filter(ce -> !getMessage(ce).isEmpty())
      .map(CallExpression::calleeSymbol)
      .map(Symbol::fullyQualifiedName)
      .filter(PandasReadNoDataTypeCheck::isPandasReadCall)
      .isPresent();
  }

  private static boolean isPandasReadCall(String fqn) {
    return "pandas.read_csv".equals(fqn) || "pandas.read_table".equals(fqn);
  }

  private static String getMessage(CallExpression ce) {
    return Optional.ofNullable(ce.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .map(name -> String.format("%s '%s'.", MESSAGE, name))
      .orElse("");
  }
}
