package org.sonar.python.checks;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;

@Rule(key="S2142")
public class IgnoredSystemExitOrKeyboardInterruptCheck extends PythonSubscriptionCheck {

  private static final List<String> INTERRUPT_EXCEPTIONS = Arrays.asList("SystemExit", "KeyboardInterrupt");
  private static final String BASE_EXCEPTION_NAME = "BaseException";

  private static final String MESSAGE_NOT_RERAISED_CAUGHT_EXCEPTION = "Reraise this exception to stop the application as the user expects";
  private static final String MESSAGE_BARE_EXCEPT = "Specify an exception class to catch or reraise the exception";
  private static final String MESSAGE_NOT_RERAISED_BASE_EXCEPTION = "Catch a more specific exception or reraise the exception";

  /**
   * Checks that a given expression is re-raised eventually.
   */
  private static class ExceptionReRaiseCheckVisitor extends BaseTreeVisitor {
    private Symbol exceptionInstance;
    private boolean isReRaised;

    public ExceptionReRaiseCheckVisitor(Symbol exceptionInstance) {
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
  }

  private Symbol findExceptionInstanceSymbol(Expression exceptionInstance) {
    Symbol exceptionInstanceSymbol = null;
    if (exceptionInstance instanceof HasSymbol) {
      exceptionInstanceSymbol = ((HasSymbol) exceptionInstance).symbol();
    }
    return exceptionInstanceSymbol;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.TRY_STMT, ctx -> {
      TryStatement tryStatement = (TryStatement) ctx.syntaxNode();

      Set<String> handledInterrupts = new HashSet<>();
      for (ExceptClause exceptClause : tryStatement.exceptClauses()) {
        Expression caughtException = exceptClause.exception();
        String caughtExceptionName = null;
        if (caughtException instanceof HasSymbol) {
          Symbol exceptionSymbol = ((HasSymbol) caughtException).symbol();
          if (exceptionSymbol != null) {
            caughtExceptionName = exceptionSymbol.fullyQualifiedName();
          }
        }

        Expression exceptionInstance = exceptClause.exceptionInstance();

        Symbol exceptionInstanceSymbol = findExceptionInstanceSymbol(exceptionInstance);
        ExceptionReRaiseCheckVisitor visitor = new ExceptionReRaiseCheckVisitor(exceptionInstanceSymbol);
        exceptClause.accept(visitor);

        if (visitor.isReRaised) {
          // The exception has been re-raised in the except clause
          continue;
        }

        if (caughtException == null) {
          // This is a bare except clause, the code should have handled interrupts at this point.
          if (handledInterrupts.size() != INTERRUPT_EXCEPTIONS.size()) {
            ctx.addIssue(exceptClause.exceptKeyword(), MESSAGE_BARE_EXCEPT);
            break;
          }
          handledInterrupts.addAll(INTERRUPT_EXCEPTIONS);
        } else if (INTERRUPT_EXCEPTIONS.contains(caughtExceptionName)) {
          ctx.addIssue(caughtException, MESSAGE_NOT_RERAISED_CAUGHT_EXCEPTION);
          handledInterrupts.add(caughtExceptionName);
        } else if (BASE_EXCEPTION_NAME.equals(caughtExceptionName)) {
          ctx.addIssue(caughtException, MESSAGE_NOT_RERAISED_BASE_EXCEPTION);
          handledInterrupts.addAll(INTERRUPT_EXCEPTIONS);
        }
      }
    });
  }
}
