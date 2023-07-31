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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S2876")
public class IterMethodReturnTypeCheck extends PythonSubscriptionCheck {
  private static final String INVALID_RETURN_VALUE_MESSAGE = "Return an object complying with iterator protocol.";
  private static final String NO_RETURN_STMTS_MESSAGE = INVALID_RETURN_VALUE_MESSAGE
    + " Consider explicitly raising a NotImplementedError if this class is not (yet) meant to support this method.";
  private static final String COROUTINE_METHOD_MESSAGE = INVALID_RETURN_VALUE_MESSAGE + " The method can not be a coroutine and have the `async` keyword.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> checkClassDefinition(ctx, (ClassDef) ctx.syntaxNode()));
  }

  private static void checkClassDefinition(SubscriptionContext ctx, ClassDef classDef) {
    for (var methodDef : TreeUtils.topLevelFunctionDefs(classDef)) {
      checkFunctionDefinition(ctx, methodDef, CheckUtils.mustBeAProtocolLike(classDef));
    }
  }

  private static void checkFunctionDefinition(SubscriptionContext ctx, FunctionDef funDef, boolean classIsProtocolLike) {
    String funNameString = funDef.name().name();
    if (!"__iter__".equals(funNameString)) {
      return;
    }

    ReturnCheckUtils.addIssueIfAsync(ctx, funDef, COROUTINE_METHOD_MESSAGE);

    var returnStmtCollector = ReturnCheckUtils.ReturnStmtCollector.collect(funDef);

    // // If there are yield keywords, then the method always returns a generator that supports the iterator protocol
    if (returnStmtCollector.containsYield()) {
      return;
    }

    List<ReturnStatement> returnStmts = returnStmtCollector.getReturnStmts();
    // If there are no return statements, we trigger this rule since this effectively means that the method is returning `None`.
    // However, there are exceptions to this:
    // * if the method raises exceptions, then it likely does not return a value on purpose
    // * if the method is marked as abstract or the surrounding class is a Protocol, then it is likely not implemented on purpose
    if (returnStmts.isEmpty() &&
      !returnStmtCollector.raisesExceptions() &&
      !CheckUtils.isAbstract(funDef) &&
      !classIsProtocolLike) {
      ctx.addIssue(funDef.defKeyword(), funDef.colon(), NO_RETURN_STMTS_MESSAGE);
      return;
    }

    for (ReturnStatement returnStmt : returnStmts) {
      checkReturnStmt(ctx, returnStmt);
    }
  }

  private static void checkReturnStmt(SubscriptionContext ctx, ReturnStatement returnStmt) {
    List<Expression> returnedExpressions = returnStmt.expressions();
    if (returnedExpressions.isEmpty()) {
      ctx.addIssue(returnStmt.returnKeyword(), INVALID_RETURN_VALUE_MESSAGE);
      return;
    }

    InferredType returnStmtType = returnStmt.returnValueType();
    if (!returnStmtType.canHaveMember("__iter__") ||
      !returnStmtType.canHaveMember("__next__")) {
      ReturnCheckUtils.addIssueOnReturnedExpressions(ctx, returnStmt, INVALID_RETURN_VALUE_MESSAGE);
    }
  }
}
