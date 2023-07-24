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
import java.util.List;
import java.util.Map;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TupleImpl;
import org.sonar.python.types.InferredTypes;

@Rule(key = "S6658")
public class SpecialMethodReturnTypeCheck extends PythonSubscriptionCheck {
  /**
   * Stores the return types expected for specific method names.
   * Each pair is annotated with a citation of the wording of the official documentation used to describe the expected type:
   *   https://docs.python.org/3/reference/datamodel.html#special-method-names
   *   https://docs.python.org/3/library/pickle.html#pickling-class-instances
   *
   * However, in practice, the python interpreter is not as strict as the wording in the documentation.
   * For instance, {@code __str__(self)} is also allowed to return a sub-type of {@code str} without throwing a type error.
   */
  private static final Map<String, String> METHOD_TO_RETURN_TYPE = Map.of(
    // wording: "should return False or True"
    "__bool__", BuiltinTypes.BOOL,
    // wording: "must be an integer"
    "__index__", BuiltinTypes.INT,
    // wording: "must be a string object"
    "__repr__", BuiltinTypes.STR,
    // wording: "must be a string object"
    "__str__", BuiltinTypes.STR,
    // wording: "should return a bytes object"
    "__bytes__", BuiltinTypes.BYTES,
    // wording: "should return an integer"
    "__hash__", BuiltinTypes.INT,
    // wording: "return value must be a string object"
    "__format__", BuiltinTypes.STR,
    // wording: "must return a tuple"
    "__getnewargs__", BuiltinTypes.TUPLE,
    // wording: "must return a pair (args, kwargs) where args is a tuple of positional arguments and kwargs a dictionary of named arguments"
    "__getnewargs_ex__", BuiltinTypes.TUPLE);

  private static final String INVALID_RETURN_TYPE_MESSAGE = "Return a value of type `%s` here.";
  private static final String INVALID_RETURN_TYPE_MESSAGE_NO_LOCATION = "Return a value of type `%s` in this method.";
  private static final String COROUTINE_METHOD_MESSAGE = INVALID_RETURN_TYPE_MESSAGE_NO_LOCATION + " The method can not be a coroutine and have the `async` keyword.";
  private static final String YIELD_ELEMENT_MESSAGE = INVALID_RETURN_TYPE_MESSAGE_NO_LOCATION + " The method can not be a generator and contain `yield` expressions.";
  private static final String INVALID_GETNEWARGSEX_TUPLE_MESSAGE = String.format(INVALID_RETURN_TYPE_MESSAGE, "tuple[tuple, dict]");
  private static final String INVALID_GETNEWARGSEX_ELEMENT_COUNT_MESSAGE = INVALID_GETNEWARGSEX_TUPLE_MESSAGE
    + " A tuple of two elements was expected but found tuple with %d element(s).";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> checkFunctionDefinition(ctx, (FunctionDef) ctx.syntaxNode()));
  }

  private static void checkFunctionDefinition(SubscriptionContext ctx, FunctionDef funDef) {
    final String funNameString = funDef.name().name();

    final String expectedReturnType = METHOD_TO_RETURN_TYPE.get(funNameString);
    if (expectedReturnType == null) {
      return;
    }

    checkForAsync(ctx, funDef, expectedReturnType);

    final ReturnStmtCollector returnStmtCollector = collectReturnStmts(funDef);
    final List<Token> yieldKeywords = returnStmtCollector.getYieldKeywords();
    for (final Token yieldKeyword : yieldKeywords) {
      ctx.addIssue(yieldKeyword, String.format(YIELD_ELEMENT_MESSAGE, expectedReturnType));
    }

    final List<ReturnStatement> returnStmts = returnStmtCollector.getReturnStmts();
    if (returnStmts.isEmpty() && yieldKeywords.isEmpty()) {
      ctx.addIssue(funDef.defKeyword(), funDef.colon(), String.format(INVALID_RETURN_TYPE_MESSAGE_NO_LOCATION, expectedReturnType));
      return;
    }

    for (final ReturnStatement returnStmt : returnStmts) {
      checkReturnStmt(ctx, funNameString, expectedReturnType, returnStmt);
    }
  }

  private static void checkForAsync(SubscriptionContext ctx, FunctionDef funDef, String expectedReturnType) {
    final Token asyncKeyword = funDef.asyncKeyword();
    if (asyncKeyword != null) {
      ctx.addIssue(asyncKeyword, String.format(COROUTINE_METHOD_MESSAGE, expectedReturnType));
    }
  }

  /**
   * {@code checkReturnStmt} inspects the expressions contained in a return statement against the given {@code expectedReturnType}.
   * Some additional checks are performed if {@code methodName} is {@code "__getnewargs_ex__"}.
   *
   * To avoid triggering too many false positives, we perform rather weak checks using a negation of {@code canBeOrExtend}.
   * That is, if there is some path to a return statement such that the returned expression can be subtype of the expected type, then we
   * do not raise an issue.
   * Conversely, we raise an issue only if there is no way the returned expression could be (a subtype of) the expected type.
   * (But we do not check the feasibility of paths.)
   *
   * Let us take a look at some relevant examples for illustration:
   *
   * <pre>
   * class BoolMethodCheck:
   *   def __init__(self, condition):
   *     self.condition = condition
   *
   *   def __bool__(self):
   *     return self.condition
   * </pre>
   *
   * Here, {@code condition} can be anything and therefore technically it could not be a boolean.
   * However, the user may only ever create instances of {@code BoolMethodCheck} where {@code self.condition} is a boolean.
   * Hence, we do not report a rule violation in such cases as it might lead to too many FPs.
   *
   * The following case involves a Union type:
   *
   * <pre>
   * def string_unknown() -> str:
   *   ...
   * class BoolMethodCheck:
   *   def __bool__(self):
   *     x = string_unknown()
   *     if x == 'True':
   *       x = True
   *     elif x == 'False':
   *       x = False
   *   return x
   * </pre>
   *
   * Here, the type of {@code x} is identified to be {@code Union[bool, str]} (simplified).
   * In theory, this code may return a string but likely not in practice, leading to FPs.
   * Hence, we can not raise issues just because a returned union type contains a type which is not a subtype of the expected type.
   */
  private static void checkReturnStmt(SubscriptionContext ctx, String methodName, String expectedReturnType, ReturnStatement returnStmt) {
    final List<Expression> returnedExpressions = returnStmt.expressions();
    if (returnedExpressions.isEmpty()) {
      ctx.addIssue(returnStmt.returnKeyword(), String.format(INVALID_RETURN_TYPE_MESSAGE_NO_LOCATION, expectedReturnType));
      return;
    }

    final InferredType returnStmtType = getReturnStmtType(returnStmt);
    if (!returnStmtType.canBeOrExtend(expectedReturnType)) {
      addIssueOnReturnedExpressions(ctx, returnStmt, String.format(INVALID_RETURN_TYPE_MESSAGE, expectedReturnType));
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
      final Expression firstExpression = returnedExpressions.get(0);
      if (firstExpression instanceof TupleImpl) {
        // If a single expression is being returned, and it is a tuple, we directly inspect its elements:
        returnedExpressions = ((TupleImpl) firstExpression).elements();
        numReturnedExpressions = returnedExpressions.size();
      } else {
        // If there is only one expression being returned, and it is not a tuple expression, then
        // we can not tell if it is a compliant tuple without a more sophisticated analysis for tracking values.
        // Hence, we abort in this case.

        return;
      }
    }

    if (numReturnedExpressions != 2) {
      addIssueOnReturnedExpressions(ctx, returnStatement, String.format(INVALID_GETNEWARGSEX_ELEMENT_COUNT_MESSAGE, numReturnedExpressions));
      return;
    }

    // Exactly two expressions are being returned
    final Expression firstElement = returnedExpressions.get(0);
    final Expression secondElement = returnedExpressions.get(1);

    if (!firstElement.type().canBeOrExtend(BuiltinTypes.TUPLE) ||
      !secondElement.type().canBeOrExtend(BuiltinTypes.DICT)) {
      ctx.addIssue(firstElement.firstToken(), secondElement.lastToken(), INVALID_GETNEWARGSEX_TUPLE_MESSAGE);
    }
  }

  private static InferredType getReturnStmtType(ReturnStatement returnStatement) {
    final List<Expression> returnedExpressions = returnStatement.expressions();

    if (returnedExpressions.isEmpty()) {
      return InferredTypes.NONE;
    }

    if (returnedExpressions.size() == 1) {
      return returnedExpressions.get(0).type();
    }

    return InferredTypes.TUPLE;
  }

  private static void addIssueOnReturnedExpressions(SubscriptionContext ctx, ReturnStatement returnStatement, String message) {
    final List<Expression> returnedExpressions = returnStatement.expressions();

    if (returnedExpressions.isEmpty()) {
      ctx.addIssue(returnStatement.returnKeyword(), message);
    } else {
      final Token firstExpressionToken = returnedExpressions.get(0).firstToken();
      final Token lastExpressionToken = returnedExpressions.get(returnedExpressions.size() - 1).lastToken();

      ctx.addIssue(firstExpressionToken, lastExpressionToken, message);
    }
  }

  private static ReturnStmtCollector collectReturnStmts(FunctionDef funDef) {
    final ReturnStmtCollector returnExpressionCollector = new ReturnStmtCollector();
    funDef.body().accept(returnExpressionCollector);

    return returnExpressionCollector;
  }

  private static class ReturnStmtCollector extends BaseTreeVisitor {
    private final List<ReturnStatement> returnStmts = new ArrayList<>();
    private List<Token> yieldKeywords = new ArrayList<>();

    public List<ReturnStatement> getReturnStmts() {
      return returnStmts;
    }

    public List<Token> getYieldKeywords() {
      return yieldKeywords;
    }

    @Override
    public void visitReturnStatement(ReturnStatement pyReturnStatementTree) {
      returnStmts.add(pyReturnStatementTree);
    }

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      // We do not visit nested function definitions as they may contain irrelevant return statements
    }

    @Override
    public void visitYieldStatement(YieldStatement pyYieldStatementTree) {
      yieldKeywords.add(pyYieldStatementTree.yieldExpression().yieldKeyword());
    }

    @Override
    public void visitYieldExpression(YieldExpression pyYieldExpressionTree) {
      yieldKeywords.add(pyYieldExpressionTree.yieldKeyword());
    }

    @Override
    public void visitLambda(LambdaExpression pyLambdaExpressionTree) {
      // We do not visit nested lambda definitions as they may contain irrelevant yield expressions
    }
  }
}
