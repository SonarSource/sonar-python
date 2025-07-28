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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7515")
public class AsyncWithContextManagerCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use \"async with\" instead of \"with\" for this asynchronous context manager.";
  private static final String SECONDARY_TYPE_DEFINITION = "This context manager implements the async context manager protocol.";
  private static final String SECONDARY_MESSAGE = "This function is async.";

  private TypeCheckBuilder hasAsyncEnter;
  private TypeCheckBuilder hasAsyncExit;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::setupTypeChecker);
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_STMT, this::checkWithStatement);
  }

  private void setupTypeChecker(SubscriptionContext ctx) {
    hasAsyncEnter = ctx.typeChecker().typeCheckBuilder().hasMember("__aenter__");
    hasAsyncExit = ctx.typeChecker().typeCheckBuilder().hasMember("__aexit__");
  }

  private void checkWithStatement(SubscriptionContext ctx) {
    WithStatement withStatement = (WithStatement) ctx.syntaxNode();
    if (withStatement.isAsync()) {
      return;
    }
    var asyncToken = TreeUtils.asyncTokenOfEnclosingFunction(withStatement).orElse(null);
    if (asyncToken == null) {
      return;
    }
    for (WithItem item : withStatement.withItems()) {
      Expression contextManager = item.test();
      PythonType contextManagerType = contextManager.typeV2();
      if (implementsAsyncContextManagerProtocol(contextManagerType)) {
        PreciseIssue issue = ctx.addIssue(withStatement.withKeyword(), MESSAGE)
          .secondary(asyncToken, SECONDARY_MESSAGE);
        contextManagerType.definitionLocation().ifPresent(
          location -> issue.secondary(location, SECONDARY_TYPE_DEFINITION)
        );
        break;
      }
    }
  }

  private boolean implementsAsyncContextManagerProtocol(PythonType type) {
    return hasAsyncEnter.check(type) == TriBool.TRUE && hasAsyncExit.check(type) == TriBool.TRUE;
  }
}
