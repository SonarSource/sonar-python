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

import java.util.Optional;
import java.util.Set;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7513")
public class TaskGroupNurseryUsedOnlyOnceCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Replace the %s with a direct function call when it only ever spawns one task.";
  private static final String SECONDARY_MESSAGE = "Only task created here";

  private TypeCheckBuilder asyncioTaskGroupTypeChecker;
  private TypeCheckBuilder trioNurseryTypeChecker;
  private TypeCheckBuilder anyioTaskGroupTypeChecker;
  private static final Set<String> START_METHOD_NAMES = Set.of("start_soon", "start", "create_task");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initTypeCheckers);
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_STMT, this::checkWithStatement);
  }

  private void initTypeCheckers(SubscriptionContext ctx) {
    asyncioTaskGroupTypeChecker = ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName("asyncio.TaskGroup");
    trioNurseryTypeChecker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("trio.open_nursery");
    anyioTaskGroupTypeChecker = ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName("anyio.create_task_group");
  }

  private void checkWithStatement(SubscriptionContext ctx) {
    var withStmt = (WithStatement) ctx.syntaxNode();
    if (!withStmt.isAsync()) {
      return;
    }
    for (var item : withStmt.withItems()) {
      handleWithItem(ctx, item, withStmt);
    }
  }

  private void handleWithItem(SubscriptionContext ctx, WithItem item, WithStatement withStmt) {
    var test = item.test();
    if (!(test instanceof CallExpression callExpr) || isTaskGroupOrNurseryCall(callExpr).isEmpty()) {
      return;
    }

    var aliasExpr = item.expression();
    if (!(aliasExpr instanceof Name aliasNameTree)) {
      return;
    }

    var aliasSym = aliasNameTree.symbolV2();
    if (aliasSym == null) {
      return;
    }

    shouldRaise(aliasSym, withStmt)
      .ifPresent(spawnCall -> ctx.addIssue(aliasExpr, String.format(MESSAGE, isTaskGroupOrNurseryCall(callExpr).get())).secondary(spawnCall, SECONDARY_MESSAGE));
  }

  private Optional<String> isTaskGroupOrNurseryCall(CallExpression callExpr) {
    var calleeType = callExpr.callee().typeV2();
    if (asyncioTaskGroupTypeChecker.check(calleeType) == TriBool.TRUE || anyioTaskGroupTypeChecker.check(calleeType) == TriBool.TRUE) {
      return Optional.of("TaskGroup");
    }
    if (trioNurseryTypeChecker.check(calleeType) == TriBool.TRUE) {
      return Optional.of("Nursery");
    }
    return Optional.empty();
  }

  private static boolean isSpawnCall(QualifiedExpression qualifiedExpression) {
    return START_METHOD_NAMES.contains(qualifiedExpression.name().name());
  }

  private static Optional<Tree> shouldRaise(SymbolV2 aliasSym, WithStatement withStmt) {
    var usagesInWithScope = aliasSym.usages().stream()
      .filter(u -> !u.isBindingUsage())
      .filter(u -> TreeUtils.firstAncestor(u.tree(), t -> t == withStmt) != null)
      .toList();

    Tree spawnCall = null;
    for (var usage : usagesInWithScope) {
      var parent = usage.tree().parent();

      if (parent instanceof QualifiedExpression qe && isSpawnCall(qe)) {
        if (spawnCall != null || isInsideLoop(qe)) {
          return Optional.empty();
        }
        spawnCall = qe;
      } else {
        return Optional.empty();
      }
    }

    return Optional.ofNullable(spawnCall);
  }

  private static boolean isInsideLoop(Tree tree) {
    return TreeUtils.firstAncestorOfKind(tree, Tree.Kind.FOR_STMT, Tree.Kind.WHILE_STMT) != null;
  }
}
