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
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.BreakStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ContinueStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7514")
public class ControlFlowInTaskGroupCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Refactor the block to eliminate this \"%s\" statement.";
  private static final String SECONDARY_MESSAGE = "This is an async task group context.";
  private List<TypeCheckBuilder> taskGroupTypeChecks = new ArrayList<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::setupTypeChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_STMT, this::checkWithStatement);
  }

  private void setupTypeChecks(SubscriptionContext ctx) {
    taskGroupTypeChecks = new ArrayList<>();
    taskGroupTypeChecks.add(ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName("asyncio.TaskGroup"));
    taskGroupTypeChecks.add(ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("trio.open_nursery"));
    taskGroupTypeChecks.add(ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName("anyio.create_task_group"));
  }

  private void checkWithStatement(SubscriptionContext ctx) {
    WithStatement withStatement = (WithStatement) ctx.syntaxNode();
    if (!withStatement.isAsync()) {
      return;
    }

    Optional<Expression> taskGroupItem = withStatement.withItems().stream()
      .map(WithItem::test)
      .filter(this::isTaskGroupOrNursery).findFirst();
    taskGroupItem.ifPresent(t -> {
        ControlFlowVisitor visitor = new ControlFlowVisitor(ctx, t);
        withStatement.statements().accept(visitor);
      }
    );
  }

  private boolean isTaskGroupOrNursery(Expression expression) {
    boolean result = taskGroupTypeChecks.stream()
      .anyMatch(typeCheck -> typeCheck.check(expression.typeV2()) == TriBool.TRUE);
    if (!result && expression instanceof CallExpression callExpression) {
      result = taskGroupTypeChecks.stream()
        .anyMatch(typeCheck -> typeCheck.check(callExpression.callee().typeV2()) == TriBool.TRUE);
    }
    return result;
  }

  private static class ControlFlowVisitor extends BaseTreeVisitor {
    private final SubscriptionContext ctx;
    private final Expression taskGroupItem;

    public ControlFlowVisitor(SubscriptionContext ctx, Expression taskGroupItem) {
      this.ctx = ctx;
      this.taskGroupItem = taskGroupItem;
    }

    @Override
    public void visitReturnStatement(ReturnStatement returnStatement) {
      ctx.addIssue(returnStatement, String.format(MESSAGE, "return"))
        .secondary(taskGroupItem, SECONDARY_MESSAGE);
    }

    @Override
    public void visitBreakStatement(BreakStatement breakStatement) {
      ctx.addIssue(breakStatement, String.format(MESSAGE, "break"))
        .secondary(taskGroupItem, SECONDARY_MESSAGE);
    }

    @Override
    public void visitContinueStatement(ContinueStatement continueStatement) {
      ctx.addIssue(continueStatement, String.format(MESSAGE, "continue"))
        .secondary(taskGroupItem, SECONDARY_MESSAGE);
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // Skip nested functions
    }
  }
}
