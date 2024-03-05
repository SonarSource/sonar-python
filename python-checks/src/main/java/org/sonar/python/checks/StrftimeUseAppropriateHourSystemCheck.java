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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6883")
public class StrftimeUseAppropriateHourSystemCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Use %I (12-hour clock) or %H (24-hour clock) without %p (AM/PM).";
  public static final String MESSAGE_12_HOURS = "Use %I (12-hour clock) with %p (AM/PM).";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, StrftimeUseAppropriateHourSystemCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol == null) {
      return;
    }
    String fullyQualifiedName = calleeSymbol.fullyQualifiedName();
    if (!"datetime.time.strftime".equals(fullyQualifiedName)) {
      return;
    }

    RegularArgument formatArgument = TreeUtils.nthArgumentOrKeyword(0, "format", callExpression.arguments());
    if (formatArgument == null) {
      return;
    }
    if (!formatArgument.expression().is(Tree.Kind.STRING_LITERAL)) {
      return;
    }
    StringLiteral formatString = (StringLiteral) formatArgument.expression();
    String format = formatString.trimmedQuotesValue();
    if (format.contains("%H") && format.contains("%p")) {
      context.addIssue(formatString, MESSAGE);
    } else if (format.contains("%I") && !format.contains("%p")) {
      context.addIssue(formatString, MESSAGE_12_HOURS);
    }
  }
}
