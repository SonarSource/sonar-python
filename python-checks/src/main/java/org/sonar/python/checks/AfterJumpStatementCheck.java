/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1763")
public class AfterJumpStatementCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, ctx ->
      {
        FileInput fileInput = (FileInput) ctx.syntaxNode();
        checkCfg(ControlFlowGraph.build(fileInput, ctx.pythonFile()), ctx, fileInput.statements());
      }
    );
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx ->
      {
        FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
        checkCfg(ControlFlowGraph.build(functionDef, ctx.pythonFile()), ctx, functionDef.body());
      }
    );

  }

  private static void checkCfg(@Nullable ControlFlowGraph cfg, SubscriptionContext ctx, @Nullable StatementList body) {
    if (cfg == null || body == null) {
      return;
    }
    TryStatementVisitor tryStatementVisitor = new TryStatementVisitor();
    body.accept(tryStatementVisitor);
    // to avoid FP in the CFG, we exclude try statement containing jumps
    if (tryStatementVisitor.hasTryStatementContainingJump) {
      return;
    }
    for (CfgBlock cfgBlock : cfg.blocks()) {
      if (cfgBlock.predecessors().isEmpty() && !cfgBlock.equals(cfg.start()) && !cfgBlock.elements().isEmpty()) {
        Tree firstElement = cfgBlock.elements().get(0);
        Tree lastElement = cfgBlock.elements().get(cfgBlock.elements().size() - 1);
        PreciseIssue issue = ctx.addIssue(firstElement.firstToken(), lastElement.lastToken(), "Delete this unreachable code or refactor the code to make it reachable.");
        cfg.blocks().stream()
          .filter(block -> cfgBlock.equals(block.syntacticSuccessor()))
          .map(block -> block.elements().get(block.elements().size() - 1))
          .forEach(jumpStatement -> issue.secondary(jumpStatement, "Statement exiting the current code block."));
      }
    }
  }

  private static class TryStatementVisitor extends BaseTreeVisitor {
    boolean hasTryStatementContainingJump = false;
    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // don't go inside functions
    }

    @Override
    public void visitTryStatement(TryStatement tryStatement) {
      if (!hasTryStatementContainingJump) {
        hasTryStatementContainingJump = TreeUtils.hasDescendant(tryStatement, tree -> tree.is(Kind.BREAK_STMT, Kind.CONTINUE_STMT, Kind.RETURN_STMT));
      }
    }
  }
}
