/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks;

import javax.annotation.Nullable;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7497")
public class CancellationReraisedInAsyncCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Ensure that the %s exception is re-raised after your cleanup code.";

  private static final String ASYNCIO_CANCELLED_ERROR = "asyncio.CancelledError";
  private static final String TRIO_CANCELLED = "trio.Cancelled";
  private static final String ANYIO_CANCELLED = "anyio.get_cancelled_exc_class";
  private static final String MESSAGE_SECONDARY = "This function is async.";

  private TypeCheckMap<String> cancelledExceptionTypeCheckMap;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, this::initializeChecksForFile);
    context.registerSyntaxNodeConsumer(Kind.TRY_STMT, this::checkTryStatement);
  }

  private void initializeChecksForFile(SubscriptionContext ctx) {
    cancelledExceptionTypeCheckMap = new TypeCheckMap<>();
    cancelledExceptionTypeCheckMap.put(ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName(ASYNCIO_CANCELLED_ERROR), ASYNCIO_CANCELLED_ERROR);
    cancelledExceptionTypeCheckMap.put(ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(TRIO_CANCELLED), TRIO_CANCELLED);
    cancelledExceptionTypeCheckMap.put(ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName(ANYIO_CANCELLED), "cancellation");

  }

  private void checkTryStatement(SubscriptionContext context) {
    var tryStatement = (TryStatement) context.syntaxNode();
    var enclosingAsyncToken = TreeUtils.asyncTokenOfEnclosingFunction(tryStatement).orElse(null);

    if (enclosingAsyncToken == null) {
      return;
    }

    tryStatement.exceptClauses().forEach(exceptClause -> checkExceptClause(context, exceptClause, enclosingAsyncToken));
  }

  private void checkExceptClause(SubscriptionContext subscriptionContext, ExceptClause exceptClause, Token enclosingAsyncToken) {
    var exception = exceptClause.exception();
    if (exception == null) {
      return;
    }
    String isOffendingException = isOffendingException(exception);
    if (isOffendingException.isEmpty()) {
      return;
    }

    var isCompliant = true;
    var hasTopLevelPass = exceptClause.body().statements().stream().anyMatch(s -> s.is(Kind.PASS_STMT));
    if (hasTopLevelPass) {
      isCompliant = false;
    } else {
      var complianceChecker = new ComplianceChecker(exceptClause);
      exceptClause.accept(complianceChecker);
      isCompliant = complianceChecker.isCompliant();
    }

    if (!isCompliant) {
      subscriptionContext.addIssue(
          exception,
          String.format(MESSAGE, isOffendingException))
        .secondary(enclosingAsyncToken, MESSAGE_SECONDARY);
    }
  }

  private String isOffendingException(Expression exception) {
    var isOffendingTypeOpt = cancelledExceptionTypeCheckMap.getOptionalForType(exception.typeV2());
    if (isOffendingTypeOpt.isPresent()) {
      return isOffendingTypeOpt.get();
    }
    if (exception instanceof CallExpression callExpression) {
      var result = cancelledExceptionTypeCheckMap.getOptionalForType(callExpression.callee().typeV2());
      if (result.isPresent()) {
        return result.get();
      }
    }
    return "";

  }

  static class ComplianceChecker extends BaseTreeVisitor {
    private final ExceptClause exceptClause;
    private boolean hasRaiseStatement = false;
    private boolean isCompliant = true;

    ComplianceChecker(ExceptClause exceptClause) {
      this.exceptClause = exceptClause;
    }

    boolean isCompliant() {
      // If we never found a raise statement, it's not compliant
      return isCompliant && hasRaiseStatement;
    }

    private static boolean isReRaise(RaiseStatement raiseStatement, ExceptClause exceptClause) {
      var isBareRaise = raiseStatement.expressions().isEmpty();
      var exceptionInstanceName = TreeUtils.toOptionalInstanceOf(Name.class, exceptClause.exceptionInstance());

      var isReRaise = raiseStatement.expressions().size() == 1 && exceptionInstanceName.isPresent()
        && raiseStatement.expressions().get(0).is(Kind.NAME)
        && ((Name) raiseStatement.expressions().get(0)).name().equals(exceptionInstanceName.get().name());

      return isBareRaise || isReRaise;
    }

    @Override
    public void visitRaiseStatement(RaiseStatement raiseStatement) {
      hasRaiseStatement = true;

      // Check if this is a proper re-raise
      if (!isReRaise(raiseStatement, exceptClause)) {
        isCompliant = false;
        return;
      }

      super.visitRaiseStatement(raiseStatement);
    }

    @Override
    public void visitReturnStatement(ReturnStatement returnStatement) {
      isCompliant = false;
    }

    @Override
    protected void scan(@Nullable Tree tree) {
      if (!isCompliant) {
        return;
      }
      super.scan(tree);
    }
  }
}
