/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S2876")
public class IterMethodReturnTypeCheck extends PythonSubscriptionCheck {
  private static final String INVALID_RETURN_VALUE_MESSAGE = "Return an object complying with iterator protocol.";
  private static final String NO_RETURN_STMTS_MESSAGE = INVALID_RETURN_VALUE_MESSAGE
    + " Consider explicitly raising a NotImplementedError if this class is not (yet) meant to support this method.";
  private static final String COROUTINE_METHOD_MESSAGE = INVALID_RETURN_VALUE_MESSAGE + " The method can not be a coroutine and have the `async` keyword.";

  private static final List<String> REQUIRED_ITERATOR_METHODS = List.of("__iter__", "__next__");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> checkClassDefinition(ctx, (ClassDef) ctx.syntaxNode()));
  }

  private static void checkClassDefinition(SubscriptionContext ctx, ClassDef classDef) {
    for (var methodDef : TreeUtils.topLevelFunctionDefs(classDef)) {
      checkFunctionDefinition(ctx, classDef, methodDef, CheckUtils.mustBeAProtocolLike(classDef));
    }
  }

  private static void checkFunctionDefinition(SubscriptionContext ctx, ClassDef classDef, FunctionDef funDef, boolean classIsProtocolLike) {
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
      checkReturnStmt(ctx, classDef, funDef, returnStmt);
    }
  }

  private static void checkReturnStmt(SubscriptionContext ctx, ClassDef classDef, FunctionDef functionDef, ReturnStatement returnStmt) {
    List<Expression> returnedExpressions = returnStmt.expressions();
    if (returnedExpressions.isEmpty()) {
      ctx.addIssue(returnStmt.returnKeyword(), INVALID_RETURN_VALUE_MESSAGE);
      return;
    }

    // The type analysis will report ANY for the type of "self".
    // This is technically the case, but in practice, self should almost always be an instance of the surrounding class if the associated
    // symbol is the same as the first parameter of the method.
    // Hence, to avoid FNs for methods returning self, we should report an issue if the surrounding class does not support the iterator
    // protocol.
    // Especially because it is quite common that iterator objects return self in __iter__,
    // see https://docs.python.org/3/library/stdtypes.html#iterator.__iter__
    if (returnsJustSelf(functionDef, returnedExpressions)) {
      var classSymbol = TreeUtils.getClassSymbolFromDef(classDef);
      if (classSymbol != null) {
        if (REQUIRED_ITERATOR_METHODS.stream().anyMatch(method -> !classSymbol.canHaveMember(method))) {
          ReturnCheckUtils.addIssueOnReturnedExpressions(ctx, returnStmt, INVALID_RETURN_VALUE_MESSAGE);
        }
        return;
      }
    }

    // if the returned expression is not `self`, we just rely on the type analysis.
    InferredType returnStmtType = returnStmt.returnValueType();
    if (REQUIRED_ITERATOR_METHODS.stream().anyMatch(method -> !returnStmtType.canHaveMember(method))) {
      ReturnCheckUtils.addIssueOnReturnedExpressions(ctx, returnStmt, INVALID_RETURN_VALUE_MESSAGE);
    }
  }

  private static boolean returnsJustSelf(FunctionDef funDef, List<Expression> returnedExpressions) {
    if (returnedExpressions.size() != 1) {
      return false;
    }

    var firstReturnedExpression = returnedExpressions.get(0);
    if (!CheckUtils.isSelf(firstReturnedExpression)) {
      return false;
    }

    var returnedSelfSymbol = ((Name) firstReturnedExpression).symbol();
    if (returnedSelfSymbol == null) {
      return false;
    }

    var selfParameterSymbol = CheckUtils.findFirstParameterSymbol(funDef);
    if (selfParameterSymbol == null) {
      return false;
    }

    return returnedSelfSymbol == selfParameterSymbol;
  }
}
