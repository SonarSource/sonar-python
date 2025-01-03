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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.CfgBranchingBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1751")
public class LoopExecutingAtMostOnceCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx ->
      checkCfg(ControlFlowGraph.build((FunctionDef) ctx.syntaxNode(), ctx.pythonFile()), ctx)
    );
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, ctx ->
      checkCfg(ControlFlowGraph.build((FileInput) ctx.syntaxNode(), ctx.pythonFile()), ctx)
    );
  }

  private static void checkCfg(@Nullable ControlFlowGraph cfg, SubscriptionContext ctx) {
    if (cfg == null) {
      return;
    }
    cfg.blocks().stream()
      .filter(CfgBranchingBlock.class::isInstance)
      .map(CfgBranchingBlock.class::cast)
      .filter(b -> b.branchingTree().is(Kind.WHILE_STMT))
      .forEach(b -> checkLoop(b, ctx));
  }

  private static void checkLoop(CfgBranchingBlock loopBlock, SubscriptionContext ctx) {
    Tree loop = loopBlock.branchingTree();
    // try to find path in CFG from trueSuccessor of loopBlock to the loopBlock
    // if such path exists then loop can be executed multiple times
    Deque<CfgBlock> workList = new ArrayDeque<>();
    workList.add(loopBlock.trueSuccessor());
    Set<CfgBlock> seen = new HashSet<>();
    List<Token> jumps = new ArrayList<>();
    while (!workList.isEmpty()) {
      CfgBlock b = workList.pop();
      if (b.successors().contains(loopBlock)) {
        return;
      }
      if (seen.add(b)) {
        if (b.syntacticSuccessor() != null && !breakOfInnerLoop(b, loopBlock)) {
          jumps.add(b.elements().get(b.elements().size() - 1).firstToken());
        }
        b.successors().stream()
          // consider only paths within the loop body
          .filter(succ -> blockInsideLoop(succ, loop))
          .forEach(workList::push);
      }
    }
    if (TreeUtils.hasDescendant(loop, t -> t.is(Kind.TRY_STMT))) {
      return;
    }
    PreciseIssue issue = ctx.addIssue(loop.firstToken(), "Refactor this loop to do more than one iteration.");
    jumps.forEach(j -> issue.secondary(j, "The loop stops here."));
  }

  private static boolean breakOfInnerLoop(CfgBlock block, CfgBranchingBlock loopBlock) {
    WhileStatement loop = (WhileStatement) loopBlock.branchingTree();
    CfgBlock breakTarget = loop.elseClause() == null
      ? loopBlock.falseSuccessor()
      // assumption: elseBlock is always a simple block, hence having only one successor
      : loopBlock.falseSuccessor().successors().iterator().next();
    Tree jumpStatement = block.elements().get(block.elements().size() - 1);
    return jumpStatement.is(Kind.BREAK_STMT) && block.successors().stream().noneMatch(b -> b == breakTarget);
  }

  private static boolean blockInsideLoop(CfgBlock block, Tree loop) {
    List<Tree> elements = block.elements();
    return elements.isEmpty() || TreeUtils.firstAncestor(elements.get(0), tree -> tree == loop) != null;
  }

}
