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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6735")
public class PandasAddMergeParametersCheck extends PythonSubscriptionCheck {

  private static final List<String> messages = List.of(
    "The '%s' parameter of the merge should be specified.",
    "The '%s' and '%s' parameters of the merge should be specified.",
    "The '%s', '%s' and '%s' parameters of the merge should be specified.");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, PandasAddMergeParametersCheck::verifyMergeCallParameters);
  }

  private static void verifyMergeCallParameters(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(callExpression.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter("pandas.merge"::equals)
      .ifPresent(fqn -> missingArguments(ctx, callExpression));
  }

  private static void missingArguments(SubscriptionContext ctx, CallExpression callExpression) {
    List<String> parameters = new ArrayList<>();
    if (TreeUtils.argumentByKeyword("how", callExpression.arguments()) == null) {
      parameters.add("how");
    }
    if (TreeUtils.argumentByKeyword("on", callExpression.arguments()) == null) {
      parameters.add("on");
    }
    if (TreeUtils.argumentByKeyword("validate", callExpression.arguments()) == null) {
      parameters.add("validate");
    }
    if (!parameters.isEmpty()) {
      ctx.addIssue(callExpression, String.format(messages.get(parameters.size() - 1), parameters.toArray()));
    }

  }
}
