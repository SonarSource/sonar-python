/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import java.util.Map;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.tree.TupleImpl;

@Rule(key = "S935")
public class SpecialMethodReturnTypeCheck extends PythonSubscriptionCheck {
  /**
   * Stores the return types expected for specific method names as specified here:
   *   <a href="https://docs.python.org/3/reference/datamodel.html#special-method-names">Special Method Names</a>
   *   <a href="https://docs.python.org/3/library/pickle.html#pickling-class-instances">__getnewargs__, __getnewargs_ex__ Documentation</a>
   * <p>
   * (In practice, the python interpreter is not as strict as the wording of the documentation.
   * For instance, {@code __str__(self)} is allowed to return a subtype of {@code str} without throwing a type error.
   * We respect the behaviour of the python interpreter in this regard.)
   */
  private static final Map<String, String> METHOD_TO_RETURN_TYPE = Map.of(
    "__bool__", BuiltinTypes.BOOL,
    "__index__", BuiltinTypes.INT,
    "__repr__", BuiltinTypes.STR,
    "__str__", BuiltinTypes.STR,
    "__bytes__", BuiltinTypes.BYTES,
    "__hash__", BuiltinTypes.INT,
    "__format__", BuiltinTypes.STR,
    "__getnewargs__", BuiltinTypes.TUPLE,
    "__getnewargs_ex__", BuiltinTypes.TUPLE);

  private static final String INVALID_RETURN_TYPE_MESSAGE = "Return a value of type `%s` here.";
  private static final String INVALID_RETURN_TYPE_MESSAGE_NO_LOCATION = "Return a value of type `%s` in this method.";
  private static final String NO_RETURN_STMTS_MESSAGE = INVALID_RETURN_TYPE_MESSAGE_NO_LOCATION
    + " Consider explicitly raising a TypeError if this class is not meant to support this method.";
  private static final String COROUTINE_METHOD_MESSAGE = INVALID_RETURN_TYPE_MESSAGE_NO_LOCATION + " The method can not be a coroutine and have the `async` keyword.";
  private static final String GENERATOR_METHOD_MESSAGE = INVALID_RETURN_TYPE_MESSAGE_NO_LOCATION + " The method can not be a generator and contain `yield` expressions.";
  private static final String INVALID_GETNEWARGSEX_TUPLE_MESSAGE = String.format(INVALID_RETURN_TYPE_MESSAGE, "tuple[tuple, dict]");
  private static final String INVALID_GETNEWARGSEX_ELEMENT_COUNT_MESSAGE = INVALID_GETNEWARGSEX_TUPLE_MESSAGE
    + " A tuple of two elements was expected but found tuple with %d element(s).";

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
    String expectedReturnType = METHOD_TO_RETURN_TYPE.get(funNameString);
    if (expectedReturnType == null) {
      return;
    }

    ReturnCheckUtils.addIssueIfAsync(ctx, funDef, String.format(COROUTINE_METHOD_MESSAGE, expectedReturnType));

    ReturnCheckUtils.ReturnStmtCollector returnStmtCollector = ReturnCheckUtils.ReturnStmtCollector.collect(funDef);
    List<Token> yieldKeywords = returnStmtCollector.getYieldKeywords();
    for (Token yieldKeyword : yieldKeywords) {
      ctx.addIssue(yieldKeyword, String.format(GENERATOR_METHOD_MESSAGE, expectedReturnType));
    }

    List<ReturnStatement> returnStmts = returnStmtCollector.getReturnStmts();
    // If there are no return statements, we trigger this rule since this effectively means that the method is returning `None`.
    // However, there are exceptions to this:
    // * if there are yield keywords, these already trigger the rule, so there is no reason to add even more issues
    // * if the method raises exceptions, then it likely does not return a value on purpose, see docstring of `raisesExceptions`
    // * if the method is marked as abstract or the surrounding class is a Protocol, then it is likely not implemented on purpose
    if (returnStmts.isEmpty() &&
      yieldKeywords.isEmpty() &&
      !returnStmtCollector.raisesExceptions() &&
      !CheckUtils.isAbstract(funDef) &&
      !classIsProtocolLike) {
      ctx.addIssue(funDef.defKeyword(), funDef.colon(), String.format(NO_RETURN_STMTS_MESSAGE, expectedReturnType));
      return;
    }

    for (ReturnStatement returnStmt : returnStmts) {
      checkReturnStmt(ctx, funNameString, expectedReturnType, returnStmt);
    }
  }

  /**
   * {@code checkReturnStmt} inspects the expressions contained in a return statement against the given {@code expectedReturnType}.
   * Some additional checks are performed if {@code methodName} is {@code "__getnewargs_ex__"}.
   */
  private static void checkReturnStmt(SubscriptionContext ctx, String methodName, String expectedReturnType, ReturnStatement returnStmt) {
    List<Expression> returnedExpressions = returnStmt.expressions();
    if (returnedExpressions.isEmpty()) {
      ctx.addIssue(returnStmt.returnKeyword(), String.format(INVALID_RETURN_TYPE_MESSAGE_NO_LOCATION, expectedReturnType));
      return;
    }

    InferredType returnStmtType = returnStmt.returnValueType();
    // To avoid FPs, we raise an issue only if there is no way a returned expression could be (a subtype of) the expected type.
    if (!returnStmtType.canBeOrExtend(expectedReturnType)) {
      ReturnCheckUtils.addIssueOnReturnedExpressions(ctx, returnStmt, String.format(INVALID_RETURN_TYPE_MESSAGE, expectedReturnType));
      return;
    }

    if ("__getnewargs_ex__".equals(methodName)) {
      isGetNewArgsExCompliant(ctx, returnStmt);
    }
  }

  private static void isGetNewArgsExCompliant(SubscriptionContext ctx, ReturnStatement returnStatement) {
    List<Expression> returnedExpressions = returnStatement.expressions();
    int numReturnedExpressions = returnedExpressions.size();

    // If there is only one expression being returned, it might be a tuple wrapped in parentheses.
    // I.e.
    //
    // return a, b
    //
    // and
    //
    // return (a, b)
    //
    // both return a tuple of two elements.
    //
    // We check for the second case and unwrap it:
    if (numReturnedExpressions == 1) {
      Expression firstExpression = returnedExpressions.get(0);
      if (firstExpression instanceof TupleImpl tupleImpl) {
        // If a single expression is being returned, and it is a tuple, we directly inspect its elements:
        returnedExpressions = tupleImpl.elements();
        numReturnedExpressions = returnedExpressions.size();
      } else {
        // If there is only one expression being returned, and it is not a tuple expression, then
        // we can not tell if it is a compliant tuple without a more sophisticated analysis for tracking values.
        // Hence, we abort in this case.

        return;
      }
    }

    if (numReturnedExpressions != 2) {
      ReturnCheckUtils.addIssueOnReturnedExpressions(ctx, returnStatement, String.format(INVALID_GETNEWARGSEX_ELEMENT_COUNT_MESSAGE, numReturnedExpressions));
      return;
    }

    // Exactly two expressions are being returned
    Expression firstElement = returnedExpressions.get(0);
    Expression secondElement = returnedExpressions.get(1);

    if (!firstElement.type().canBeOrExtend(BuiltinTypes.TUPLE) ||
      !secondElement.type().canBeOrExtend(BuiltinTypes.DICT)) {
      ctx.addIssue(firstElement.firstToken(), secondElement.lastToken(), INVALID_GETNEWARGSEX_TUPLE_MESSAGE);
    }
  }
}
