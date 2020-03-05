/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
import org.sonar.plugins.python.api.tree.Tuple;

@Rule(key = "S5713")
public class ChildAndParentExceptionCaughtCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.EXCEPT_CLAUSE, ctx -> {
      ExceptClause exceptClause = (ExceptClause) ctx.syntaxNode();
      Map<String, List<Expression>> caughtExceptionsByFQN = new HashMap<>();
      Expression exceptionExpression = exceptClause.exception();
      if (exceptionExpression == null || !exceptionExpression.is(Tree.Kind.TUPLE)) {
        return;
      }
      Tuple exceptionsTuple = (Tuple) exceptionExpression;
      exceptionsTuple.elements().forEach(e -> addExceptionExpression(e, caughtExceptionsByFQN));
      checkCaughtExceptions(ctx, caughtExceptionsByFQN);
    });
  }

  private static void checkCaughtExceptions(SubscriptionContext ctx, Map<String, List<Expression>> caughtExceptionsByFQN) {
    caughtExceptionsByFQN.forEach((fullyQualifiedName, caughtExceptionsWithSameFQN) -> {
      Expression currentException = caughtExceptionsWithSameFQN.get(0);
      ClassSymbol currentSymbol = (ClassSymbol) ((HasSymbol) currentException).symbol();
      if (caughtExceptionsWithSameFQN.size() > 1) {
        PreciseIssue issue = ctx.addIssue(currentException, "Remove this duplicate Exception class.");
        caughtExceptionsWithSameFQN.stream().filter(e -> e != currentException).forEach(e -> issue.secondary(e, null));
      }
      PreciseIssue issue = null;
      for (List<Expression> otherCaughtExceptions : caughtExceptionsByFQN.values()) {
        if (otherCaughtExceptions == caughtExceptionsWithSameFQN) {
          continue;
        }
        Expression comparedException = otherCaughtExceptions.get(0);
        ClassSymbol comparedSymbol = (ClassSymbol) ((HasSymbol) comparedException).symbol();
        if (currentSymbol.isOrExtends(comparedSymbol.fullyQualifiedName())) {
          if (issue == null) {
            issue = ctx.addIssue(currentException, "Remove this redundant Exception class; it derives from another which is already caught.");
            addSecondaryLocations(issue, otherCaughtExceptions);
          } else {
            addSecondaryLocations(issue, otherCaughtExceptions);
          }
        }
      }
    });
  }

  private static void addExceptionExpression(Expression exceptionExpression, Map<String, List<Expression>> caughtExceptionsByFQN) {
    if (exceptionExpression instanceof HasSymbol) {
      Symbol symbol = ((HasSymbol) exceptionExpression).symbol();
      if (symbol != null && symbol.kind().equals(Symbol.Kind.CLASS)) {
        ClassSymbol classSymbol = (ClassSymbol) symbol;
        List<Expression> exceptions = caughtExceptionsByFQN.getOrDefault(classSymbol.fullyQualifiedName(), new ArrayList<>());
        exceptions.add(exceptionExpression);
        caughtExceptionsByFQN.putIfAbsent(classSymbol.fullyQualifiedName(), exceptions);
      }
    }
  }

  private static void addSecondaryLocations(PreciseIssue issue, List<Expression> others) {
    for (Expression other : others) {
      issue.secondary(other, null);
    }
  }
}
