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
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7502")
public class AsyncioTaskNotStoredCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Save this task in a variable to prevent premature garbage collection.";
  private static final Set<String> TASK_CREATION_FUNCTIONS = Set.of("asyncio.create_task", "asyncio.ensure_future");

  private final List<TypeCheckBuilder> asyncioTaskTypeChecks = new ArrayList<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::setupTypeChecker);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpression);
  }

  private void setupTypeChecker(SubscriptionContext ctx) {
    asyncioTaskTypeChecks.clear();
    TASK_CREATION_FUNCTIONS.forEach(fqn -> asyncioTaskTypeChecks.add(ctx.typeChecker().typeCheckBuilder().isTypeWithName(fqn)));
  }

  private void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Expression callee = callExpression.callee();
    PythonType calleeType = callee.typeV2();
    if (asyncioTaskTypeChecks.stream().noneMatch(t -> t.check(calleeType) == TriBool.TRUE)) {
      return;
    }
    if (TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.ASSIGNMENT_EXPRESSION, Tree.Kind.ANNOTATED_ASSIGNMENT, Tree.Kind.CALL_EXPR,
      Tree.Kind.RETURN_STMT) == null) {
      ctx.addIssue(callee, MESSAGE);
    }
  }
}
