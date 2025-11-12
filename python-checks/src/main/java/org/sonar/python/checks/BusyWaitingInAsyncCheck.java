/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
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

import java.util.Optional;

import javax.annotation.Nullable;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AwaitExpression;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7484")
public class BusyWaitingInAsyncCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Refactor this loop to use an `Event` instead of polling with `sleep`.";
  private static final String SECONDARY_MESSAGE = "This function is async.";
  private static final String POLLING_MESSAGE = "Polling happens here.";

  private TypeCheckMap<Object> asyncSleepTypeChecks;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, this::initializeTypeCheckMap);
    context.registerSyntaxNodeConsumer(Kind.WHILE_STMT, this::checkWhileStatement);
  }

  private void initializeTypeCheckMap(SubscriptionContext ctx) {
    var object = new Object();
    asyncSleepTypeChecks = new TypeCheckMap<>();
    asyncSleepTypeChecks.put(ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName("asyncio.sleep"), object);
    asyncSleepTypeChecks.put(ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("trio.sleep"), object);
    asyncSleepTypeChecks.put(ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName("anyio.sleep"), object);
  }

  private void checkWhileStatement(SubscriptionContext ctx) {
    var whileStmt = (WhileStatement) ctx.syntaxNode();
    var enclosingFuncDef = (FunctionDef) TreeUtils.firstAncestorOfKind(whileStmt, Kind.FUNCDEF);
    var asyncToken = Optional.ofNullable(enclosingFuncDef).map(FunctionDef::asyncKeyword);
    if (asyncToken.isEmpty()) {
      return;
    }

    var sleepFinder = new SleepCallFinder(asyncSleepTypeChecks);
    whileStmt.body().accept(sleepFinder);

    if (sleepFinder.sleepAwait == null) {
      return;
    }

    var conditionChecker = new GlobalOrNonLocalNameFinder(enclosingFuncDef);
    whileStmt.condition().accept(conditionChecker);

    if (conditionChecker.foundGlobalOrNonLocal) {
      ctx.addIssue(whileStmt.condition(), MESSAGE)
        .secondary(sleepFinder.sleepAwait, POLLING_MESSAGE)
        .secondary(asyncToken.get(), SECONDARY_MESSAGE);
    }
  }

  private static class GlobalOrNonLocalNameFinder extends BaseTreeVisitor {
    boolean foundGlobalOrNonLocal = false;
    private final FunctionDef whileLoopFunction;

    GlobalOrNonLocalNameFinder(FunctionDef whileLoopFunction) {
      this.whileLoopFunction = whileLoopFunction;
    }

    @Override
    public void visitName(Name name) {
      if (foundGlobalOrNonLocal) {
        return;
      }
      var symbol = name.symbol();
      if (symbol != null && isSymbolDeclaredInOuterScope(symbol, whileLoopFunction)) {
        foundGlobalOrNonLocal = true;
      }
      super.visitName(name);
    }

    // Use of Symbol V1 instead of SymbolV2 because of SONARPY-2974
    private static boolean isSymbolDeclaredInOuterScope(Symbol symbol, FunctionDef currentFunctionContext) {
      var fileInput = (FileInput) TreeUtils.firstAncestorOfKind(currentFunctionContext, Kind.FILE_INPUT);
      for (var usage : symbol.usages()) {
        if (usage.kind() == Usage.Kind.ASSIGNMENT_LHS) {
          if (fileInput.globalVariables().contains(symbol)) {
            return true;
          }
          var currentFunction = (FunctionDef) TreeUtils.firstAncestorOfKind(currentFunctionContext, Kind.FUNCDEF);
          while (currentFunction != null) {
            if (currentFunction.localVariables().contains(symbol)) {
              return true;
            }
            currentFunction = (FunctionDef) TreeUtils.firstAncestorOfKind(currentFunction, Kind.FUNCDEF);
          }

        }
      }
      return false;
    }
  }

  private static class SleepCallFinder extends BaseTreeVisitor {
    Tree sleepAwait = null;
    private final TypeCheckMap<Object> asyncSleepTypeChecks;

    SleepCallFinder(TypeCheckMap<Object> asyncSleepTypeChecks) {
      this.asyncSleepTypeChecks = asyncSleepTypeChecks;
    }

    @Override
    public void visitAwaitExpression(AwaitExpression awaitExpr) {
      var expr = awaitExpr.expression();
      if (expr instanceof CallExpression callExpr && isAsyncSleepCall(callExpr.callee())) {
        sleepAwait = awaitExpr.awaitToken();
      }
      super.visitAwaitExpression(awaitExpr);
    }

    @Override
    protected void scan(@Nullable Tree tree) {
      if (sleepAwait != null) {
        return;
      }
      super.scan(tree);
    }

    private boolean isAsyncSleepCall(Expression call) {
      return asyncSleepTypeChecks.getOptionalForType(call.typeV2()).isPresent();
    }
  }
}
