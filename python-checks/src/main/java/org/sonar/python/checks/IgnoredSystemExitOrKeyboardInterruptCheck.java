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

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5754")
public class IgnoredSystemExitOrKeyboardInterruptCheck extends PythonSubscriptionCheck {

  private static final String BASE_EXCEPTION_NAME = "BaseException";
  private static final String SYSTEM_EXIT_EXCEPTION_NAME = "SystemExit";
  private static final String KEYBOARD_INTERRUPT_NAME = "KeyboardInterrupt";

  private static final List<String> INTERRUPT_EXCEPTIONS = Arrays.asList(SYSTEM_EXIT_EXCEPTION_NAME, KEYBOARD_INTERRUPT_NAME);

  private static final String MESSAGE_NOT_RERAISED_CAUGHT_EXCEPTION = "Reraise this exception to stop the application as the user expects";
  private static final String MESSAGE_BARE_EXCEPT = "Specify an exception class to catch or reraise the exception";
  private static final String MESSAGE_NOT_RERAISED_BASE_EXCEPTION = "Catch a more specific exception or reraise the exception";

  private static final String SYSTEM_EXIT_FUNCTION_NAME = "sys.exit";
  private static final String SYSTEM_EXC_INFO_NAME = "sys.exc_info";

  /**
   * Checks that a given expression is re-raised or eventually handled.
   */
  private static class ExceptionReRaiseCheckVisitor extends BaseTreeVisitor {
    private Symbol exceptionInstance;
    private boolean isReRaised;

    public ExceptionReRaiseCheckVisitor(@Nullable Symbol exceptionInstance) {
      this.exceptionInstance = exceptionInstance;
    }

    @Override
    public void visitRaiseStatement(RaiseStatement pyRaiseStatementTree) {
      if (pyRaiseStatementTree.expressions().isEmpty()) {
        // It is a bare raise, which re-raises the last encountered exception
        this.isReRaised = true;
        return;
      }

      Expression raisedException = pyRaiseStatementTree.expressions().get(0);
      if (raisedException.type().canOnlyBe(SYSTEM_EXIT_EXCEPTION_NAME)) {
        this.isReRaised = true;
      }

      if (raisedException instanceof HasSymbol) {
        Symbol symbol = ((HasSymbol) raisedException).symbol();
        if (symbol == null) {
          // The symbol is unknown, bail out
          return;
        }

        if (symbol.equals(this.exceptionInstance)) {
          this.isReRaised = true;
        }
      }
    }

    @Override
    public void visitCallExpression(CallExpression pyCallExpressionTree) {
      Symbol symbol = pyCallExpressionTree.calleeSymbol();
      if (symbol == null) {
        return;
      }

      String fqn = symbol.fullyQualifiedName();
      this.isReRaised |= SYSTEM_EXIT_FUNCTION_NAME.equals(fqn);
      this.isReRaised |= SYSTEM_EXC_INFO_NAME.equals(fqn);
    }
  }

  private static Symbol findExceptionInstanceSymbol(Expression exceptionInstance) {
    Symbol exceptionInstanceSymbol = null;
    if (exceptionInstance instanceof HasSymbol) {
      exceptionInstanceSymbol = ((HasSymbol) exceptionInstance).symbol();
    }
    return exceptionInstanceSymbol;
  }

  private static String findExceptionName(Expression exception) {
    if (exception instanceof HasSymbol) {
      Symbol exceptionSymbol = ((HasSymbol) exception).symbol();
      if (exceptionSymbol != null) {
        return exceptionSymbol.fullyQualifiedName();
      }
    }

    return null;
  }

  /**
   * Checks whether a possibly bare except clause is compliant and raises the issue if not.
   * Returns true if the except clause was bare, false otherwise.
   */
  private static boolean handlePossibleBareException(SubscriptionContext ctx, ExceptClause exceptClause, Set<String> handledInterrupts) {
    Expression exceptionExpr = exceptClause.exception();
    if (exceptionExpr != null) {
      return false;
    }

    ExceptionReRaiseCheckVisitor visitor = new ExceptionReRaiseCheckVisitor(null);
    exceptClause.accept(visitor);
    if (!visitor.isReRaised && !handledInterrupts.containsAll(INTERRUPT_EXCEPTIONS)) {
      ctx.addIssue(exceptClause.exceptKeyword(), MESSAGE_BARE_EXCEPT);
    }

    return true;
  }

  private static void handleCaughtException(SubscriptionContext ctx, Expression caughtException, Symbol exceptionInstanceSymbol,
    Tree exceptionBody, Set<String> handledInterrupts) {
    String caughtExceptionName = findExceptionName(caughtException);
    if (caughtExceptionName == null) {
      // The caught exception name is unknown, just skip to the next clause.
      return;
    }

    handledInterrupts.add(caughtExceptionName);

    ExceptionReRaiseCheckVisitor visitor = new ExceptionReRaiseCheckVisitor(exceptionInstanceSymbol);
    exceptionBody.accept(visitor);

    if (visitor.isReRaised) {
      // The exception has been handled in the except clause
      return;
    }

    // If the KeyboardInterrupt error was handled in any way (even possibly ignored), do not raise an issue.
    if (SYSTEM_EXIT_EXCEPTION_NAME.equals(caughtExceptionName)) {
      ctx.addIssue(caughtException, MESSAGE_NOT_RERAISED_CAUGHT_EXCEPTION);
    } else if (BASE_EXCEPTION_NAME.equals(caughtExceptionName) && !handledInterrupts.containsAll(INTERRUPT_EXCEPTIONS)) {
      ctx.addIssue(caughtException, MESSAGE_NOT_RERAISED_BASE_EXCEPTION);
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.TRY_STMT, ctx -> {
      TryStatement tryStatement = (TryStatement) ctx.syntaxNode();
      Set<String> handledInterrupts = new HashSet<>();

      for (ExceptClause exceptClause : tryStatement.exceptClauses()) {
        Expression exceptionExpr = exceptClause.exception();
        if (handlePossibleBareException(ctx, exceptClause, handledInterrupts)) {
          break;
        }

        // Find the possible exception instance name
        Expression exceptionInstance = exceptClause.exceptionInstance();
        Symbol exceptionInstanceSymbol = findExceptionInstanceSymbol(exceptionInstance);

        List<Expression> caughtExceptions = TreeUtils.flattenTuples(exceptionExpr).collect(Collectors.toList());
        for (Expression caughtException : caughtExceptions) {
          handleCaughtException(ctx, caughtException, exceptionInstanceSymbol, exceptClause.body(), handledInterrupts);
        }
      }
    });
  }

}
