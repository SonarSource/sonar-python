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

import java.util.List;
import java.util.Objects;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;

@Rule(key = "S6887")
public class DateTimeNoPyTzTimeZoneInConstructor extends PythonSubscriptionCheck {

  private ReachingDefinitionsAnalysis reachingDefinitionsAnalysis;
  private static final String MESSAGE = "pytz.timezone should not be passed to the datetime.datetime constructor";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT,
      ctx -> reachingDefinitionsAnalysis = new ReachingDefinitionsAnalysis(ctx.pythonFile()));

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkContext);
  }

  private void checkContext(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();

    if (calleeSymbol != null && "datetime.datetime".equals(calleeSymbol.fullyQualifiedName())) {
      ArgList argList = callExpression.argumentList();
      if (argList != null) {
        List<Argument> argumentList = argList.arguments();
        argumentList.stream().filter(this::checkArgument)
          .findAny().ifPresent(argument -> context.addIssue(argument, MESSAGE));
      }
    }
  }

  private boolean checkArgument(Argument argument) {
    if (argument.is(Tree.Kind.REGULAR_ARGUMENT)) {
      RegularArgument regularArgument = ((RegularArgument) argument);
      if (regularArgument.expression().is(Tree.Kind.CALL_EXPR)) {
        CallExpression callExpression1 = (CallExpression) regularArgument.expression();
        Symbol calleeSymbol = callExpression1.calleeSymbol();
        return calleeSymbol != null && "pytz.timezone".equals(calleeSymbol.fullyQualifiedName());
      } else if (regularArgument.expression().is(Tree.Kind.NAME)) {
        return reachingDefinitionsAnalysis.valuesAtLocation((Name) regularArgument.expression()).stream()
          .filter(expression -> expression.is(Tree.Kind.CALL_EXPR))
          .map(CallExpression.class::cast)
          .map(CallExpression::calleeSymbol)
          .filter(Objects::nonNull)
          .anyMatch(symbol -> "pytz.timezone".equals(symbol.fullyQualifiedName()));
      }
    }
    return false;
  }
}
