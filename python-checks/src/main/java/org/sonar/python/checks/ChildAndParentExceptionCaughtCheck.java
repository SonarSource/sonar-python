/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5713")
public class ChildAndParentExceptionCaughtCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.EXCEPT_CLAUSE, ctx -> {
      ExceptClause exceptClause = (ExceptClause) ctx.syntaxNode();
      Map<ClassSymbol, List<Expression>> caughtExceptionsBySymbol = new HashMap<>();
      Expression exceptionExpression = exceptClause.exception();
      if (exceptionExpression == null) {
        return;
      }
      TreeUtils.flattenTuples(exceptionExpression).forEach(e -> addExceptionExpression(e, caughtExceptionsBySymbol));
      checkCaughtExceptions(ctx, caughtExceptionsBySymbol);
    });
  }

  private static void checkCaughtExceptions(SubscriptionContext ctx, Map<ClassSymbol, List<Expression>> caughtExceptionsBySymbol) {
    caughtExceptionsBySymbol.forEach((currentSymbol, caughtExceptionsWithSameSymbol) -> {
      Expression currentException = caughtExceptionsWithSameSymbol.get(0);
      if (caughtExceptionsWithSameSymbol.size() > 1) {
        PreciseIssue issue = ctx.addIssue(currentException, "Remove this duplicate Exception class.");
        caughtExceptionsWithSameSymbol.stream().skip(1).forEach(e -> issue.secondary(e, "Duplicate."));
      }
      PreciseIssue issue = null;
      for (Map.Entry<ClassSymbol, List<Expression>> otherEntry : caughtExceptionsBySymbol.entrySet()) {
        ClassSymbol comparedSymbol = otherEntry.getKey();
        if (currentSymbol != comparedSymbol && currentSymbol.isOrExtends(comparedSymbol)) {
          if (issue == null) {
            issue = ctx.addIssue(currentException, "Remove this redundant Exception class; it derives from another which is already caught.");
          }
          addSecondaryLocations(issue, otherEntry.getValue());
        }
      }
    });
  }

  private static void addExceptionExpression(Expression exceptionExpression, Map<ClassSymbol, List<Expression>> caughtExceptionsByFQN) {
    if (exceptionExpression instanceof HasSymbol) {
      Symbol symbol = ((HasSymbol) exceptionExpression).symbol();
      if (symbol != null && symbol.kind().equals(Symbol.Kind.CLASS)) {
        ClassSymbol classSymbol = (ClassSymbol) symbol;
        caughtExceptionsByFQN.computeIfAbsent(classSymbol, k -> new ArrayList<>()).add(exceptionExpression);
      }
    }
  }

  private static void addSecondaryLocations(PreciseIssue issue, List<Expression> others) {
    for (Expression other : others) {
      issue.secondary(other, "Parent class.");
    }
  }
}
