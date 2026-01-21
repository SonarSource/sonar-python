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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8405")
public class TestClientContentParameterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use \"content\" parameter instead of \"data\" for bytes or text.";
  private static final String STARLETTE_TEST_CLIENT_FQN = "starlette.testclient.TestClient";

  private static final Set<String> HTTP_METHOD_NAMES = Set.of(
    "get", "post", "put", "delete", "patch", "head", "options", "request");

  private static final TypeMatcher IS_TEST_CLIENT_METHOD = TypeMatchers.any(
    HTTP_METHOD_NAMES.stream()
      .map(name -> STARLETTE_TEST_CLIENT_FQN + "." + name)
      .map(TypeMatchers::isType));

  private static final TypeMatcher IS_STRING = TypeMatchers.isObjectOfType("str");

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, TestClientContentParameterCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext ctx) {
    CallExpression call = (CallExpression) ctx.syntaxNode();

    if (!isTestClientHttpMethodCall(call, ctx)) {
      return;
    }

    RegularArgument dataArg = TreeUtils.argumentByKeyword("data", call.arguments());
    if (dataArg == null) {
      return;
    }

    Expression dataValue = dataArg.expression();

    if (isBytesOrStrLiteral(dataValue, ctx)) {
      Name keyword = dataArg.keywordArgument();
      if (keyword != null) {
        ctx.addIssue(keyword, MESSAGE);
      }
    }
  }

  private static boolean isTestClientHttpMethodCall(CallExpression call, SubscriptionContext ctx) {
    Expression callee = call.callee();
    return IS_TEST_CLIENT_METHOD.isTrueFor(callee, ctx);
  }

  private static boolean isBytesOrStrLiteral(Expression expr, SubscriptionContext ctx) {
    return IS_STRING.isTrueFor(expr, ctx);
  }
}
