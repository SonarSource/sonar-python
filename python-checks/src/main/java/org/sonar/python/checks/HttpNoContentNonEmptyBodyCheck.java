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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8400")
public class HttpNoContentNonEmptyBodyCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Return an empty body for this endpoint returning 204 status.";

  private static final TypeMatcher FASTAPI_RESPONSE_INSTANCE = TypeMatchers.isObjectOfType("fastapi.Response");
  private static final TypeMatcher NONE_TYPE = TypeMatchers.isObjectOfType("NoneType");

  private static final TypeMatcher FASTAPI_METHODS_MATCHER = TypeMatchers.any(
    TypeMatchers.withFQN("fastapi.applications.FastAPI.get"),
    TypeMatchers.withFQN("fastapi.applications.FastAPI.post"),
    TypeMatchers.withFQN("fastapi.applications.FastAPI.put"),
    TypeMatchers.withFQN("fastapi.applications.FastAPI.delete"),
    TypeMatchers.withFQN("fastapi.applications.FastAPI.patch"),
    TypeMatchers.withFQN("fastapi.applications.FastAPI.options"),
    TypeMatchers.withFQN("fastapi.applications.FastAPI.head"),
    TypeMatchers.withFQN("fastapi.applications.FastAPI.trace"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, HttpNoContentNonEmptyBodyCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();

    if (!isFastApiEndpointWithNoContentStatus(ctx, functionDef)) {
      return;
    }

    findProblematicReturns(ctx, functionDef);
  }

  private static boolean isFastApiEndpointWithNoContentStatus(SubscriptionContext ctx, FunctionDef functionDef) {
    for (Decorator decorator : functionDef.decorators()) {
      Expression decoratorExpression = decorator.expression();

      if (decoratorExpression instanceof CallExpression callExpr) {
        if (!FASTAPI_METHODS_MATCHER.isTrueFor(callExpr.callee(), ctx)) {
          continue;
        }

        RegularArgument statusCodeArg = TreeUtils.argumentByKeyword("status_code", callExpr.arguments());
        if (statusCodeArg != null) {
          Expression statusValue = statusCodeArg.expression();
          if (isNoContentStatusValue(statusValue)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  private static boolean isNoContentStatusValue(Expression expr) {
    if (expr.is(Tree.Kind.NUMERIC_LITERAL)) {
      String value = expr.firstToken().value();
      return "204".equals(value);
    }
    return false;
  }

  private static void findProblematicReturns(SubscriptionContext ctx, FunctionDef functionDef) {
    List<ReturnStatement> allReturns = new ArrayList<>();
    collectReturnStatements(functionDef.body(), allReturns);

    for (ReturnStatement returnStmt : allReturns) {
      ValidationResult validationResult = isValidReturnStatement(ctx, returnStmt);
      if (!validationResult.isValid) {
        var issue = ctx.addIssue(returnStmt, MESSAGE);

        for (Tree secondaryLocation : validationResult.secondaryLocations) {
          issue.secondary(secondaryLocation, "Response is assigned here");
        }
      }
    }
  }

  private record ValidationResult(boolean isValid, List<Tree> secondaryLocations) {
    ValidationResult(boolean isValid) {
      this(isValid, new ArrayList<>());
    }
  }

  private static void collectReturnStatements(Tree tree, List<ReturnStatement> returns) {
    if (tree.is(Tree.Kind.RETURN_STMT)) {
      returns.add((ReturnStatement) tree);
      return;
    }

    if (tree.is(Tree.Kind.FUNCDEF)) {
      // Do not traverse into nested functions
      return;
    }
    tree.children().forEach(child -> collectReturnStatements(child, returns));
  }

  private static ValidationResult isValidReturnStatement(SubscriptionContext ctx, ReturnStatement returnStmt) {
    List<Expression> expressions = returnStmt.expressions();

    if (expressions.isEmpty()) {
      return new ValidationResult(true);
    }

    if (expressions.size() == 1) {
      Expression returnValue = expressions.get(0);

      if (returnValue.is(Tree.Kind.NONE)) {
        return new ValidationResult(true);
      }

      if (NONE_TYPE.isTrueFor(returnValue, ctx)) {
        return new ValidationResult(true);
      }

      // Check for Response(status_code=204) or Response(status_code=204, content="")
      return isValidResponseObject(ctx, returnValue);
    }

    // Any other return value is non-compliant
    return new ValidationResult(false);
  }

  private static ValidationResult isValidResponseObject(SubscriptionContext ctx, Expression expr) {
    if (!FASTAPI_RESPONSE_INSTANCE.isTrueFor(expr, ctx)) {
      return new ValidationResult(false);
    }

    List<Tree> secondaryLocations = new ArrayList<>();

    if (expr instanceof Name name) {
      var assignedValues = ctx.valuesAtLocation(name);

      boolean anyInvalid = false;
      for (Expression assignedValue : assignedValues) {
        if (assignedValue instanceof CallExpression callExpr && isInvalidResponseCall(callExpr)) {
          // Check if this Response has invalid arguments
          anyInvalid = true;
          secondaryLocations.add(assignedValue);
        }
      }

      if (anyInvalid) {
        return new ValidationResult(false, secondaryLocations);
      }
    } else if (expr instanceof CallExpression callExpr && isInvalidResponseCall(callExpr)) {
      // Direct Response constructor call
      return new ValidationResult(false);
    }

    return new ValidationResult(true);
  }

  private static boolean isInvalidResponseCall(CallExpression callExpr) {
    RegularArgument statusCodeArg = TreeUtils.argumentByKeyword("status_code", callExpr.arguments());
    if (statusCodeArg != null) {
      Expression statusValue = statusCodeArg.expression();
      if (!isNoContentStatusValue(statusValue)) {
        return true;
      }
    }

    RegularArgument contentArg = TreeUtils.argumentByKeyword("content", callExpr.arguments());
    if (contentArg != null) {
      Expression contentValue = contentArg.expression();
      StringLiteral stringLiteral = Expressions.extractStringLiteral(contentValue);
      if (stringLiteral == null) {
        return true;
      }
      return !stringLiteral.trimmedQuotesValue().isEmpty();
    }

    return false;
  }
}
