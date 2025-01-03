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

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.EllipsisExpression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

/**
 * Note that implementation differs from AbstractOneStatementPerLineCheck due to Python specifics
 */
@Rule(key = OneStatementPerLineCheck.CHECK_KEY)
public class OneStatementPerLineCheck extends PythonSubscriptionCheck {

  public static final String CHECK_KEY = "OneStatementPerLine";
  private final Map<Integer, AtomicInteger> statementsPerLine = new HashMap<>();
  private SubscriptionContext subscriptionContext;
  private static final Set<Tree.Kind> kinds = EnumSet.of(Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.COMPOUND_ASSIGNMENT, Tree.Kind.EXPRESSION_STMT,
    Tree.Kind.IMPORT_NAME, Tree.Kind.IMPORT_FROM, Tree.Kind.CONTINUE_STMT, Tree.Kind.BREAK_STMT, Tree.Kind.YIELD_STMT, Tree.Kind.RETURN_STMT, Tree.Kind.PRINT_STMT,
    Tree.Kind.PASS_STMT, Tree.Kind.FOR_STMT, Tree.Kind.WHILE_STMT, Tree.Kind.IF_STMT, Tree.Kind.ELSE_CLAUSE, Tree.Kind.RAISE_STMT, Tree.Kind.TRY_STMT, Tree.Kind.EXCEPT_CLAUSE,
    Tree.Kind.EXEC_STMT, Tree.Kind.ASSERT_STMT, Tree.Kind.DEL_STMT, Tree.Kind.GLOBAL_STMT, Tree.Kind.CLASSDEF, Tree.Kind.FUNCDEF);
  private static final Map<Tree.Kind, Predicate<Tree>> IS_EXCEPTION_PREDICATES_MAP = Map.of(
    Tree.Kind.EXPRESSION_STMT, OneStatementPerLineCheck::isDummyImplementationEllipsis
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      statementsPerLine.clear();
      this.subscriptionContext = ctx;
    });

    kinds.forEach(k -> context.registerSyntaxNodeConsumer(k, this::checkStatement));
  }

  private void checkStatement(SubscriptionContext ctx) {
    if (IS_EXCEPTION_PREDICATES_MAP.containsKey(ctx.syntaxNode().getKind()) && IS_EXCEPTION_PREDICATES_MAP.get(ctx.syntaxNode().getKind()).test(ctx.syntaxNode())) {
      return;
    }

    int line = ctx.syntaxNode().firstToken().line();
    statementsPerLine.computeIfAbsent(line, l -> new AtomicInteger(0)).incrementAndGet();
  }

  @Override
  public void leaveFile() {
    statementsPerLine.entrySet()
      .stream()
      .filter(statementsAtLine -> statementsAtLine.getValue().get() > 1)
      .forEach(statementsAtLine -> {
        String message = String.format("At most one statement is allowed per line, but %s statements were found on this line.", statementsAtLine.getValue());
        int lineNumber = statementsAtLine.getKey();
        subscriptionContext.addLineIssue(message, lineNumber);
      });
  }

  private static boolean isDummyImplementationEllipsis(Tree t) {
    var isEllipsisExpressionOnly = TreeUtils.toOptionalInstanceOf(ExpressionStatement.class, t)
      .filter(es -> es.expressions().size() == 1)
      .map(ExpressionStatement::expressions)
      .map(expressions -> expressions.get(0))
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(EllipsisExpression.class))
      .isPresent();

    if (isEllipsisExpressionOnly) {
      // check if it is dummy function or class implementation
      return Optional.of(t)
        .map(Tree::parent)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(StatementList.class))
        .filter(body -> body.statements().size() == 1)
        .map(Tree::parent)
        .filter(p -> p.is(Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF))
        .isPresent();
    }
    return false;
  }
}
