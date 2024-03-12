/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

@Rule(key = "S6925")
public class TfGatherDeprecatedValidateIndicesCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "`validate_indices` is deprecated.";
  private static final String FQN = "tensorflow.gather";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, TfGatherDeprecatedValidateIndicesCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext context) {
    Optional.of((CallExpression) context.syntaxNode())
      .filter(callExpression -> {
        Symbol symbol = callExpression.calleeSymbol();
        return symbol != null && FQN.equals(symbol.fullyQualifiedName());
      })
      .map(callExpression -> TreeUtils.nthArgumentOrKeyword(2, "validate_indices", callExpression.arguments()))
      .ifPresent(argument -> context.addIssue(argument, MESSAGE));
  }
}
