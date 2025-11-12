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

import java.util.HashSet;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BreakStatement;
import org.sonar.plugins.python.api.tree.ContinueStatement;
import org.sonar.plugins.python.api.tree.FinallyClause;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S7932")
public class FinallyBlockExitStatementCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_FORMAT = "Remove this %s statement from the finally block.";

  private boolean isPython314OrGreater = false;
  private final Set<Tree> finallyBlocks = new HashSet<>();
  private final Set<Tree> nestedScopes = new HashSet<>();
  private final Set<Tree> loopsInFinally = new HashSet<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeState);
    context.registerSyntaxNodeConsumer(Tree.Kind.FINALLY_CLAUSE, this::enterFinallyClause);
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, this::enterNestedScope);
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, this::enterNestedScope);
    context.registerSyntaxNodeConsumer(Tree.Kind.WHILE_STMT, this::enterLoop);
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, this::enterLoop);
    context.registerSyntaxNodeConsumer(Tree.Kind.RETURN_STMT, this::checkReturnStatement);
    context.registerSyntaxNodeConsumer(Tree.Kind.BREAK_STMT, this::checkBreakStatement);
    context.registerSyntaxNodeConsumer(Tree.Kind.CONTINUE_STMT, this::checkContinueStatement);
  }

  private void initializeState(SubscriptionContext ctx) {
    isPython314OrGreater = PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(ctx.sourcePythonVersions(), PythonVersionUtils.Version.V_314);
  }

  private void enterFinallyClause(SubscriptionContext ctx) {
    FinallyClause finallyClause = (FinallyClause) ctx.syntaxNode();
    finallyBlocks.add(finallyClause);
  }

  private void enterNestedScope(SubscriptionContext ctx) {
    Tree node = ctx.syntaxNode();
    if (getEnclosingFinallyBlock(node) != null) {
      nestedScopes.add(node);
    }
  }

  private void enterLoop(SubscriptionContext ctx) {
    Tree node = ctx.syntaxNode();
    Tree finallyBlock = getEnclosingFinallyBlock(node);
    if (finallyBlock != null && !isInsideNestedClassOrFunctionWithinFinally(node, finallyBlock)) {
      loopsInFinally.add(node);
    }
  }

  private Tree getEnclosingFinallyBlock(Tree tree) {
    return TreeUtils.firstAncestor(tree, finallyBlocks::contains);
  }

  private boolean isInsideNestedClassOrFunctionWithinFinally(Tree tree, Tree finallyBlock) {
    return isInsideAncestorWithinFinally(tree, nestedScopes, finallyBlock);
  }

  private void checkReturnStatement(SubscriptionContext ctx) {
    if (!isPython314OrGreater) {
      return;
    }
    ReturnStatement returnStmt = (ReturnStatement) ctx.syntaxNode();
    Tree finallyBlock = getEnclosingFinallyBlock(returnStmt);
    if (finallyBlock != null && !isInsideNestedClassOrFunctionWithinFinally(returnStmt, finallyBlock)) {
      ctx.addIssue(returnStmt, String.format(MESSAGE_FORMAT, "return"));
    }
  }

  private void checkBreakStatement(SubscriptionContext ctx) {
    if (!isPython314OrGreater) {
      return;
    }
    BreakStatement breakStmt = (BreakStatement) ctx.syntaxNode();
    raiseIfIsNotInsideNestedClassOrFunctionOrALoop(breakStmt, "break", ctx);
  }

  private void checkContinueStatement(SubscriptionContext ctx) {
    if (!isPython314OrGreater) {
      return;
    }
    ContinueStatement continueStmt = (ContinueStatement) ctx.syntaxNode();
    raiseIfIsNotInsideNestedClassOrFunctionOrALoop(continueStmt, "continue", ctx);
  }

  private void raiseIfIsNotInsideNestedClassOrFunctionOrALoop(Tree statement, String statementType, SubscriptionContext ctx) {
    Tree finallyBlock = getEnclosingFinallyBlock(statement);
    if (finallyBlock != null && !isInsideNestedClassOrFunctionWithinFinally(statement, finallyBlock) && !isInsideLoopWithinFinally(statement, finallyBlock)) {
      ctx.addIssue(statement, String.format(MESSAGE_FORMAT, statementType));
    }
  }

  private boolean isInsideLoopWithinFinally(Tree tree, Tree finallyBlock) {
    return isInsideAncestorWithinFinally(tree, loopsInFinally, finallyBlock);
  }

  private boolean isInsideAncestorWithinFinally(Tree tree, Set<Tree> ancestors, Tree finallyBlock) {
    Tree ancestor = TreeUtils.firstAncestor(tree, ancestors::contains);
    return ancestor != null && TreeUtils.firstAncestor(ancestor, finallyBlock::equals) != null;
  }

}
