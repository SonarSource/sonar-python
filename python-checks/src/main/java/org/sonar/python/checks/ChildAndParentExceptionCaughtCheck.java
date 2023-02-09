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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
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
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5713")
public class ChildAndParentExceptionCaughtCheck extends PythonSubscriptionCheck {
  public static final String QUICK_FIX_MESSAGE = "Remove the redundant Exception";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.EXCEPT_CLAUSE, ChildAndParentExceptionCaughtCheck::checkExceptClause);
    context.registerSyntaxNodeConsumer(Tree.Kind.EXCEPT_GROUP_CLAUSE, ChildAndParentExceptionCaughtCheck::checkExceptClause);
  }

  private static void checkExceptClause(SubscriptionContext ctx) {
    ExceptClause exceptClause = (ExceptClause) ctx.syntaxNode();
    Map<ClassSymbol, List<Expression>> caughtExceptionsBySymbol = new HashMap<>();
    Expression exceptionExpression = exceptClause.exception();
    if (exceptionExpression == null) {
      return;
    }
    TreeUtils.flattenTuples(exceptionExpression).forEach(e -> addExceptionExpression(e, caughtExceptionsBySymbol));
    checkCaughtExceptions(ctx, caughtExceptionsBySymbol);
  }

  private static void checkCaughtExceptions(SubscriptionContext ctx, Map<ClassSymbol, List<Expression>> caughtExceptionsBySymbol) {
    caughtExceptionsBySymbol.forEach((currentSymbol, caughtExceptionsWithSameSymbol) -> {
      Expression currentException = caughtExceptionsWithSameSymbol.get(0);
      if (caughtExceptionsWithSameSymbol.size() > 1) {
        IssueWithQuickFix issue = (IssueWithQuickFix) ctx.addIssue(currentException, "Remove this duplicate Exception class.");
        addQuickFix(issue, currentException);
        caughtExceptionsWithSameSymbol.stream().skip(1).forEach(e -> issue.secondary(e, "Duplicate."));
      }

      var symbolsWithErrors = caughtExceptionsBySymbol.entrySet()
        .stream()
        .filter(entry -> entry.getKey() != currentSymbol && currentSymbol.isOrExtends(entry.getKey()))
        .collect(Collectors.toList());

      if (!symbolsWithErrors.isEmpty()) {
        var issue = (IssueWithQuickFix) ctx.addIssue(currentException, "Remove this redundant Exception class; it derives from another which is already caught.");
        addQuickFix(issue, currentException);

        symbolsWithErrors.stream()
          .map(Map.Entry::getValue)
          .forEach(entries -> addSecondaryLocations(issue, entries));
      }
    });
  }

  private static void addQuickFix(IssueWithQuickFix issue, Expression currentException) {
    if (currentException.parent().is(Tree.Kind.TUPLE)) {
      var quickFix = createQuickFix(currentException);
      issue.addQuickFix(quickFix);
    }
  }

  private static PythonQuickFix createQuickFix(Expression currentException) {
    PythonQuickFix.Builder builder = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE);

    var parentTuple = (Tuple) currentException.parent();

    if (parentTuple.elements().size() == 2) {
      // If there is just 2 exceptions
      // then we need just to remove everything from one which should stay till the end of the clause
      // and from the beginning of the clause till one which should stay
      parentTuple.elements().stream()
        .filter(el -> el != currentException)
        .findFirst()
        .ifPresent(toKeep -> {
          List<Tree> tupleChildren = parentTuple.children();
          var currentIndex = tupleChildren.indexOf(toKeep);

          var fromToken = tupleChildren.get(currentIndex + 1).firstToken();
          var toToken = tupleChildren.get(tupleChildren.size() - 1).lastToken();
          // toToken.column() + 1 to remove right parenthesis
          builder.addTextEdit(PythonTextEdit.removeRange(fromToken.line(), fromToken.column(), toToken.line(),  toToken.column() + 1));

          fromToken = tupleChildren.get(0).firstToken();
          toToken = toKeep.firstToken();
          builder.addTextEdit(PythonTextEdit.removeRange(fromToken.line(), fromToken.column(), toToken.line(),  toToken.column()));
        });
    } else {
      var currentIndex = parentTuple.children().indexOf(currentException);

      var firstToken = currentException.firstToken();

      // If currentException is not first one - need to remove previous comma
      if (currentIndex > 1) {
        var previous = parentTuple.children().get(currentIndex - 1);
        firstToken = previous.lastToken();
      }

      var fromLine = firstToken.line();
      var fromColumn = firstToken.column();

      var nextIndex = currentIndex + 1;
      // If currentException is first one - need to remove next comma
      if (currentIndex == 1) {
        nextIndex++;
      }
      var next = parentTuple.children().get(nextIndex);
      var nextToken = next.lastToken();
      var toLine = nextToken.line();
      var toColumn = nextToken.column();

      builder.addTextEdit(PythonTextEdit.removeRange(fromLine, fromColumn, toLine, toColumn));
    }

    return builder.build();
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
