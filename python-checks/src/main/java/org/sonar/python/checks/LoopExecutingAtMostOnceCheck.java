/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.FileInput;
import org.sonar.python.api.tree.FunctionDef;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;

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
        if (b.syntacticSuccessor() != null) {
          jumps.add(b.elements().get(b.elements().size() - 1).firstToken());
        }
        b.successors().stream()
          // consider only paths within the loop body
          .filter(succ -> blockInsideLoop(succ, loop))
          .forEach(workList::push);
      }
    }
    if (loop.descendants(Kind.TRY_STMT).findFirst().isPresent()) {
      return;
    }
    PreciseIssue issue = ctx.addIssue(loop.firstToken(), "Refactor this loop to do more than one iteration.");
    jumps.forEach(j -> issue.secondary(j, null));
  }

  private static boolean blockInsideLoop(CfgBlock block, Tree loop) {
    List<Tree> elements = block.elements();
    return elements.isEmpty() || elements.get(0).ancestors().contains(loop);
  }

}
