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
import org.sonar.plugins.python.api.tree.PassStatement;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5754")
public class IgnoredSystemExitCheck extends PythonSubscriptionCheck {

  private static final String BASE_EXCEPTION_NAME = "BaseException";
  private static final String SYSTEM_EXIT_EXCEPTION_NAME = "SystemExit";

  private static final String MESSAGE_NOT_RERAISED_CAUGHT_EXCEPTION = "Reraise this exception to stop the application as the user expects";
  private static final String MESSAGE_BARE_EXCEPT = "Specify an exception class to catch or reraise the exception";
  private static final String MESSAGE_NOT_RERAISED_BASE_EXCEPTION = "Catch a more specific exception or reraise the exception";

  private static final String SYSTEM_EXIT_FUNCTION_NAME = "sys.exit";
  private static final String SYSTEM_EXC_INFO_NAME = "sys.exc_info";
  public static final String NOT_RAISED_CAUGHT_EXCEPTION_QUICK_FIX_MESSAGE = "Replace \"pass\" statement with \"raise\"";

  /**
   * Checks that a given expression is re-raised or eventually handled.
   */
  private static class ExceptionReRaiseCheckVisitor extends BaseTreeVisitor {
    private Symbol exceptionInstance;
    private boolean isReRaised;
    private PassStatement passStatement;

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
    public void visitPassStatement(PassStatement passStatement) {
      this.passStatement = passStatement;
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

  @CheckForNull
  private static Symbol findExceptionInstanceSymbol(@Nullable Expression exceptionInstance) {
    Symbol exceptionInstanceSymbol = null;
    if (exceptionInstance instanceof HasSymbol) {
      exceptionInstanceSymbol = ((HasSymbol) exceptionInstance).symbol();
    }
    return exceptionInstanceSymbol;
  }

  @CheckForNull
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
  private static void handlePossibleBareException(SubscriptionContext ctx, ExceptClause exceptClause, boolean isSystemExitHandled) {
    ExceptionReRaiseCheckVisitor visitor = new ExceptionReRaiseCheckVisitor(null);
    exceptClause.accept(visitor);
    if (!visitor.isReRaised && !isSystemExitHandled) {
      ctx.addIssue(exceptClause.exceptKeyword(), MESSAGE_BARE_EXCEPT);
    }
  }

  /**
   * Checks whether the caught exception is properly handled.
   * @return True if the handled exception was a SystemExit, false otherwise.
   */
  private static boolean handleCaughtException(SubscriptionContext ctx, Expression caughtException, @Nullable Symbol exceptionInstanceSymbol,
    Tree exceptionBody, boolean handledSystemExit) {
    String caughtExceptionName = findExceptionName(caughtException);
    if (caughtExceptionName == null) {
      // The caught exception name is unknown, just skip to the next clause.
      return false;
    }

    ExceptionReRaiseCheckVisitor visitor = new ExceptionReRaiseCheckVisitor(exceptionInstanceSymbol);
    exceptionBody.accept(visitor);

    if (visitor.isReRaised) {
      // The exception has been handled in the except clause
      return SYSTEM_EXIT_EXCEPTION_NAME.equals(caughtExceptionName);
    }

    if (SYSTEM_EXIT_EXCEPTION_NAME.equals(caughtExceptionName)) {
      var issue = (IssueWithQuickFix) ctx.addIssue(caughtException, MESSAGE_NOT_RERAISED_CAUGHT_EXCEPTION);
      if (visitor.passStatement != null) {
        var quickFix = PythonQuickFix.newQuickFix(NOT_RAISED_CAUGHT_EXCEPTION_QUICK_FIX_MESSAGE)
          .addTextEdit(PythonTextEdit.replace(visitor.passStatement, "raise"))
          .build();
        issue.addQuickFix(quickFix);
      }

      return true;
    }

    if (BASE_EXCEPTION_NAME.equals(caughtExceptionName) && !handledSystemExit) {
      ctx.addIssue(caughtException, MESSAGE_NOT_RERAISED_BASE_EXCEPTION);
    }

    return false;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.TRY_STMT, ctx -> {
      TryStatement tryStatement = (TryStatement) ctx.syntaxNode();
      boolean isSystemExitHandled = false;

      for (ExceptClause exceptClause : tryStatement.exceptClauses()) {
        Expression exceptionExpr = exceptClause.exception();
        if (exceptionExpr == null) {
          handlePossibleBareException(ctx, exceptClause, isSystemExitHandled);
          break;
        }

        // Find the possible exception instance name
        Expression exceptionInstance = exceptClause.exceptionInstance();
        Symbol exceptionInstanceSymbol = findExceptionInstanceSymbol(exceptionInstance);

        List<Expression> caughtExceptions = TreeUtils.flattenTuples(exceptionExpr).collect(Collectors.toList());
        for (Expression caughtException : caughtExceptions) {
          isSystemExitHandled |= handleCaughtException(ctx, caughtException, exceptionInstanceSymbol, exceptClause.body(), isSystemExitHandled);
        }
      }
    });
  }

}
