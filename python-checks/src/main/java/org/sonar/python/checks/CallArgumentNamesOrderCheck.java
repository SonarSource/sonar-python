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

import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S5686")
public class CallArgumentNamesOrderCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> checkCallExpression(ctx, (CallExpression) ctx.syntaxNode()));
  }

  private static void checkCallExpression(SubscriptionContext ctx, CallExpression callExpr) {
    var parameters = functionParametersFromCallExpression(callExpr);
    if (parameters == null) {
      return;
    }
    // TODO
  }

  @CheckForNull
  private static List<AnyParameter> functionParametersFromCallExpression(CallExpression callExpr) {
    var calleeSymbol = callExpr.calleeSymbol();
    if (calleeSymbol == null) {
      return null;
    }

    if (!calleeSymbol.is(Symbol.Kind.FUNCTION)) {
      return null;
    }

    var usages = calleeSymbol.usages().stream().filter(Usage::isBindingUsage).limit(2).collect(Collectors.toUnmodifiableList());
    if (usages.size() != 1) {
      return null;
    }

    var definingUsage = usages.get(0);
    if (definingUsage.kind() != Usage.Kind.FUNC_DECLARATION) {
      return null;
    }

    var functionDefinition = definingUsage.tree().parent();
    if (!functionDefinition.is(Tree.Kind.FUNCDEF)) {
      return null;
    }

    var parameterList = ((FunctionDef) functionDefinition).parameters();
    if (parameterList == null) {
      return null;
    }

    return parameterList.all();
  }
}
