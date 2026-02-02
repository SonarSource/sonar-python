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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6863")
public class FlaskErrorHandlerStatusCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Specify an explicit HTTP status code for this error handler.";

  private static final String FLASK_APP_ERRORHANDLER_FQN = "flask.app.Flask.errorhandler";
  private static final String BLUEPRINT_ERRORHANDLER_FQN = "flask.blueprints.Blueprint.errorhandler";
  private static final String BLUEPRINT_APP_ERRORHANDLER_FQN = "flask.blueprints.Blueprint.app_errorhandler";

  private static final String JSONIFY_FQN = "flask.json.jsonify";
  private static final String RENDER_TEMPLATE_FQN = "flask.templating.render_template";
  private static final String RENDER_TEMPLATE_STRING_FQN = "flask.templating.render_template_string";
  private static final String MAKE_RESPONSE_FQN = "flask.helpers.make_response";
  private static final String RESPONSE_FQN = "flask.wrappers.Response";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, FlaskErrorHandlerStatusCheck::checkErrorHandler);
  }

  private static void checkErrorHandler(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    if (!isErrorHandlerFunction(functionDef, ctx)) {
      return;
    }
    checkFunctionReturns(ctx, functionDef);
  }

  private static void checkFunctionReturns(SubscriptionContext ctx, FunctionDef functionDef) {
    ReturnStatementVisitor visitor = new ReturnStatementVisitor(ctx);
    functionDef.body().accept(visitor);
    visitor.getProblematicReturns().forEach(returnStmt ->
      ctx.addIssue(returnStmt, MESSAGE)
    );
  }

  private static boolean isErrorHandlerFunction(FunctionDef functionDef, SubscriptionContext ctx) {
    return functionDef.decorators().stream()
      .anyMatch(decorator -> isErrorHandlerDecorator(decorator, ctx));
  }

  private static boolean isErrorHandlerDecorator(Decorator decorator, SubscriptionContext ctx) {
    Expression expression = decorator.expression();
    if (!(expression instanceof CallExpression callExpr)) {
      return false;
    }
    return TypeMatchers.any(
      TypeMatchers.isType(FLASK_APP_ERRORHANDLER_FQN),
      TypeMatchers.isType(BLUEPRINT_ERRORHANDLER_FQN),
      TypeMatchers.isType(BLUEPRINT_APP_ERRORHANDLER_FQN)
    ).isTrueFor(callExpr.callee(), ctx);
  }

  private static class ReturnStatementVisitor extends BaseTreeVisitor {
    private final SubscriptionContext ctx;
    private final List<ReturnStatement> problematicReturns = new ArrayList<>();

    public ReturnStatementVisitor(SubscriptionContext ctx) {
      this.ctx = ctx;
    }

    @Override
    public void visitReturnStatement(ReturnStatement returnStatement) {
      List<Expression> expressions = returnStatement.expressions();

      if (expressions.isEmpty()) {
        problematicReturns.add(returnStatement);
        return;
      }

      if (expressions.size() >= 2) {
        // Has status code
        return;
      }

      Expression returnValue = expressions.get(0);

      checkReturnedValue(returnStatement, returnValue);
    }

    private void checkReturnedValue(ReturnStatement returnStatement, Expression returnValue) {
      if (returnValue.is(Tree.Kind.NAME)) {
        Name nameExpr = (Name) returnValue;

        // Check if status_code is set on this variable via attribute assignment
        if (hasStatusCodeSet(nameExpr)) {
          return;
        }

        Optional<Expression> assignedValue = Expressions.singleAssignedNonNameValue(nameExpr);
        if (assignedValue.isEmpty()) {
          return;
        }

        returnValue = assignedValue.get();
        if (returnValue instanceof Tuple tuple && tuple.elements().size() >= 2) {
          return;
        }
      }

      if (returnValue instanceof CallExpression callExpr) {
        if (isProblematicFlaskFunction(callExpr)) {
          problematicReturns.add(returnStatement);
        }
        return;
      }

      problematicReturns.add(returnStatement);
    }

    private boolean isProblematicFlaskFunction(CallExpression callExpr) {
      // Functions that don't create proper Response objects
      if (TypeMatchers.any(
        TypeMatchers.isType(JSONIFY_FQN),
        TypeMatchers.isType(RENDER_TEMPLATE_FQN),
        TypeMatchers.isType(RENDER_TEMPLATE_STRING_FQN)
      ).isTrueFor(callExpr.callee(), ctx)) {
        return true;
      }

      // make_response and Response can have status as second positional arg or 'status' kwarg
      if (TypeMatchers.any(
        TypeMatchers.isType(MAKE_RESPONSE_FQN),
        TypeMatchers.isType(RESPONSE_FQN)
      ).isTrueFor(callExpr.callee(), ctx)) {
        return TreeUtils.nthArgumentOrKeyword(1, "status", callExpr.arguments()) == null;
      }

      return false;
    }

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      // Don't visit nested functions
    }

    public List<ReturnStatement> getProblematicReturns() {
      return problematicReturns;
    }

    /**
     * Check if status_code is set on the given variable via attribute assignment.
     * Example: response.status_code = 404
     */
    private static boolean hasStatusCodeSet(Name nameExpr) {
      SymbolV2 symbol = nameExpr.symbolV2();
      if (symbol == null) {
        return false;
      }

      return symbol.usages().stream()
        .map(UsageV2::tree)
        .filter(Name.class::isInstance)
        .map(Tree::parent)
        .filter(QualifiedExpression.class::isInstance)
        .map(QualifiedExpression.class::cast)
        .filter(qualifiedExpr -> "status_code".equals(qualifiedExpr.name().name()))
        .anyMatch(qualifiedExpr -> TreeUtils.firstAncestorOfKind(qualifiedExpr, Tree.Kind.ASSIGNMENT_STMT) != null);
    }
  }
}
