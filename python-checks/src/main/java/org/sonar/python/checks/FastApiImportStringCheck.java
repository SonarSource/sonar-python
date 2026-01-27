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
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8397")
public class FastApiImportStringCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_TEMPLATE = "Pass the application as an import string when using %s.";
  private static final TypeMatcher UVICORN_RUN_FUNCTION_TYPE_MATCHER = TypeMatchers.isType("uvicorn.run");
  private static final TypeMatcher IS_STRING_TYPE = TypeMatchers.isObjectOfType("str");
  private static final TypeMatcher IS_FASTAPI_APP = TypeMatchers.isObjectOfType("fastapi.applications.FastAPI");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, FastApiImportStringCheck::checkUvicornRunFunctionCalls);
  }

  private static void checkUvicornRunFunctionCalls(SubscriptionContext ctx) {
    CallExpression callExpr = ((CallExpression) ctx.syntaxNode());

    if (!UVICORN_RUN_FUNCTION_TYPE_MATCHER.isTrueFor(callExpr.callee(), ctx)) {
      return;
    }
    if (callExpr.arguments().isEmpty()) {
      return;
    }

    List<String> problematicParams = new ArrayList<>();
    boolean hasReload = false;
    boolean hasDebug = false;
    boolean hasWorkers = false;

    if (hasParameterWithTrueValue(callExpr, "reload")) {
      hasReload = true;
    }

    if (hasParameterWithTrueValue(callExpr, "debug")) {
      hasDebug = true;
    }

    if (hasWorkersParameter(callExpr)) {
      hasWorkers = true;
    }

    if (hasReload) {
      problematicParams.add("reload");
    }
    if (hasWorkers) {
      problematicParams.add("workers");
    }
    if (hasDebug) {
      problematicParams.add("debug");
    }

    if (problematicParams.isEmpty()) {
      return;
    }

    Argument firstArg = callExpr.arguments().get(0);
    if (!(firstArg instanceof RegularArgument regularArg)) {
      return;
    }

    Expression appExpr = regularArg.expression();
    if (IS_STRING_TYPE.isTrueFor(appExpr, ctx) || !IS_FASTAPI_APP.isTrueFor(appExpr, ctx)) {
      return;
    }
    String message = buildMessage(problematicParams);
    ctx.addIssue(appExpr, message);
  }

  private static boolean hasParameterWithTrueValue(CallExpression callExpr, String paramName) {
    RegularArgument arg = TreeUtils.argumentByKeyword(paramName, callExpr.arguments());
    if (arg == null) {
      return false;
    }

    Expression expr = arg.expression();
    if (!(expr instanceof Name name)) {
      return false;
    }
    if ("True".equals(expr.firstToken().value())) {
      return true;
    }
    Expression assignedValue = Expressions.singleAssignedValue(name);
    return assignedValue != null && "True".equals(assignedValue.firstToken().value());
  }

  private static boolean hasWorkersParameter(CallExpression callExpr) {
    RegularArgument arg = TreeUtils.argumentByKeyword("workers", callExpr.arguments());
    if (arg == null) {
      return false;
    }

    Expression expr = arg.expression();

    if (expr instanceof NumericLiteral numericLiteral) {
      return isGreaterThanOne(numericLiteral.valueAsString());
    }

    if (expr instanceof Name name) {
      Expression assignedValue = Expressions.singleAssignedValue(name);
      if (assignedValue instanceof NumericLiteral numericLiteral) {
        return isGreaterThanOne(numericLiteral.valueAsString());
      }
    }

    return false;
  }

  private static boolean isGreaterThanOne(String value) {
    try {
      return Integer.parseInt(value) > 1;
    } catch (NumberFormatException e) {
      return false;
    }
  }

  private static String buildMessage(List<String> problematicParams) {
    String paramsString;
    if (problematicParams.size() == 1) {
      paramsString = "'" + problematicParams.get(0) + "'";
    } else if (problematicParams.size() == 2) {
      paramsString = "'" + problematicParams.get(0) + "' and '" + problematicParams.get(1) + "'";
    } else {
      String lastParam = problematicParams.get(problematicParams.size() - 1);
      List<String> quotedParams = problematicParams.subList(0, problematicParams.size() - 1).stream()
        .map(p -> "'" + p + "'")
        .toList();
      paramsString = String.join(", ", quotedParams) + ", and '" + lastParam + "'";
    }
    return String.format(MESSAGE_TEMPLATE, paramsString);
  }
}
