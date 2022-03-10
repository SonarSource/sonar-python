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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.semantic.SymbolUtils;

@Rule(key = "S1172")
public class UnusedFunctionParameterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove the unused function parameter \"%s\".";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx -> checkFunctionParameter(ctx, ((FunctionDef) ctx.syntaxNode())));
  }

  private static void checkFunctionParameter(SubscriptionContext ctx, FunctionDef functionDef) {
    if (CheckUtils.containsCallToLocalsFunction(functionDef) || maybeOverridingMethod(functionDef) || isInterfaceMethod(functionDef)) {
      return;
    }

    functionDef.localVariables().stream()
      .filter(symbol -> !"self".equals(symbol.name()))
      .map(Symbol::usages)
      .filter(usages -> usages.size() == 1 && usages.get(0).tree().parent().is(Kind.PARAMETER))
      .map(usages -> (Parameter) usages.get(0).tree().parent())
      .forEach(param -> ctx.addIssue(param, String.format(MESSAGE, param.name().name())));
  }

  private static boolean isInterfaceMethod(FunctionDef functionDef) {
    return functionDef.body().statements().stream()
      .allMatch(statement -> statement.is(Kind.PASS_STMT, Kind.RAISE_STMT)
        || (statement.is(Kind.EXPRESSION_STMT) && isStringExpression((ExpressionStatement) statement)));
  }

  private static boolean isStringExpression(ExpressionStatement stmt) {
    return stmt.expressions().stream().allMatch(expr -> expr.is(Kind.STRING_LITERAL));
  }

  private static boolean maybeOverridingMethod(FunctionDef functionDef) {
    if (functionDef.isMethodDefinition() && !SymbolUtils.isPrivateName(functionDef.name().name())) {
      ClassDef classDef = CheckUtils.getParentClassDef(functionDef);
      return classDef != null && classDef.args() != null;
    }
    return false;
  }
}
