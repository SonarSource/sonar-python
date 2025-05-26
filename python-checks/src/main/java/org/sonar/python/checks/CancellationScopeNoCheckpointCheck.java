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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AwaitExpression;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7490")
public class CancellationScopeNoCheckpointCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add a checkpoint inside this cancellation scope.";
  private static final String SECONDARY_MESSAGE = "This checkpoint is not awaited.";

  private static final String TRIO_CHECKPOINT = "trio.lowlevel.checkpoint";
  private static final Map<String, String> TYPE_WITH_FQN_MAP = Map.of(
    "trio.CancelScope", TRIO_CHECKPOINT,
    "trio.move_on_after", TRIO_CHECKPOINT,
    "trio.move_on_at", TRIO_CHECKPOINT
  );

  private static final String ANYIO_CHECKPOINT = "anyio.lowlevel.checkpoint";
  private static final Map<String, String> TYPE_WITH_NAME_MAP = Map.of(
    "asyncio.timeout", "asyncio.sleep",
    "anyio.CancelScope", ANYIO_CHECKPOINT,
    "anyio.move_on_after", ANYIO_CHECKPOINT,
    "anyio.move_on_at", ANYIO_CHECKPOINT
  );

  private TypeCheckMap<TypeCheckBuilder> typeCheckMap;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeTypeCheckers);
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_STMT, this::checkCancellationScope);
  }

  private void initializeTypeCheckers(SubscriptionContext ctx) {
    typeCheckMap = new TypeCheckMap<>();

    TYPE_WITH_FQN_MAP.forEach((typeName, checkpointFunction) ->
      typeCheckMap.put(
        ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(typeName),
        ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(checkpointFunction)
      )
    );

    TYPE_WITH_NAME_MAP.forEach((typeName, checkpointFunction) ->
      typeCheckMap.put(
        ctx.typeChecker().typeCheckBuilder().isTypeWithName(typeName),
        ctx.typeChecker().typeCheckBuilder().isTypeWithName(checkpointFunction)
      )
    );
  }

  private void checkCancellationScope(SubscriptionContext ctx) {
    WithStatement withStatement = (WithStatement) ctx.syntaxNode();

    for (WithItem withItem : withStatement.withItems()) {
      Expression test = withItem.test();
      TypeCheckBuilder checkpointTypeChecker = getCheckpointTypeCheckerForScope(test);

      if (checkpointTypeChecker != null) {
        CheckpointVisitor visitor = new CheckpointVisitor(checkpointTypeChecker);
        withStatement.statements().accept(visitor);

        if (!visitor.hasCheckpoint()) {
          PreciseIssue issue = ctx.addIssue(test, MESSAGE);
          visitor.checkpointedTrees.forEach(t -> issue.secondary(t, SECONDARY_MESSAGE));
        }
      }
    }
  }

  @Nullable
  private TypeCheckBuilder getCheckpointTypeCheckerForScope(Expression expression) {
    if (expression.is(Tree.Kind.CALL_EXPR)) {
      CallExpression callExpression = (CallExpression) expression;
      Expression callee = callExpression.callee();
      return typeCheckMap.getOptionalForType(callee.typeV2()).orElse(null);
    }
    return typeCheckMap.getOptionalForType(expression.typeV2()).orElse(null);
  }

  private static class CheckpointVisitor extends BaseTreeVisitor {
    private final TypeCheckBuilder checkpointTypeChecker;
    private boolean checkpointFound = false;
    private List<Tree> checkpointedTrees = new ArrayList<>();

    CheckpointVisitor(TypeCheckBuilder checkpointTypeChecker) {
      this.checkpointTypeChecker = checkpointTypeChecker;
    }

    boolean hasCheckpoint() {
      return checkpointFound;
    }

    @Override
    public void visitAwaitExpression(AwaitExpression awaitExpression) {
      checkpointFound = true;
    }

    @Override
    public void visitYieldExpression(YieldExpression yieldExpression) {
      checkpointFound = true;
    }

    @Override
    public void visitForStatement(ForStatement forStatement) {
      if (forStatement.isAsync()) {
        // async for loops implicitly create checkpoints at each iteration
        checkpointFound = true;
      }
      if (!checkpointFound) {
        super.visitForStatement(forStatement);
      }
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      Expression callee = callExpression.callee();
      if (checkpointTypeChecker.check(callee.typeV2()) == TriBool.TRUE) {
        checkpointedTrees.add(callExpression);
      }
      super.visitCallExpression(callExpression);
    }

    @Override
    protected void scan(@Nullable Tree tree) {
      if (!checkpointFound && tree != null) {
        tree.accept(this);
      }
    }
  }
}
