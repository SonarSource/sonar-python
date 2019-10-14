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

import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.FileInput;
import org.sonar.python.api.tree.FunctionDef;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;

@Rule(key = "S1763")
public class AfterJumpStatementCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, ctx ->
      checkCfg(ControlFlowGraph.build((FileInput) ctx.syntaxNode(), ctx.pythonFile()), ctx)
    );
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx ->
      checkCfg(ControlFlowGraph.build((FunctionDef) ctx.syntaxNode(), ctx.pythonFile()), ctx)
    );

  }

  private static void checkCfg(@Nullable ControlFlowGraph cfg, SubscriptionContext ctx) {
    if (cfg == null) {
      return;
    }
    for (CfgBlock cfgBlock : cfg.blocks()) {
      if (cfgBlock.predecessors().isEmpty() && !cfgBlock.equals(cfg.start()) && !cfgBlock.elements().isEmpty()) {
        Tree firstElement = cfgBlock.elements().get(0);
        List<Tree> jumpStatements = cfg.blocks().stream()
          .filter(block -> cfgBlock.equals(block.syntacticSuccessor()))
          .map(block -> block.elements().get(block.elements().size() - 1))
          .collect(Collectors.toList());
        Tree lastElement = cfgBlock.elements().get(cfgBlock.elements().size() - 1);
        PreciseIssue issue = ctx.addIssue(firstElement.firstToken(), lastElement.lastToken(), "Delete this unreachable code or refactor the code to make it reachable.");
        jumpStatements.forEach(jumpStatement -> issue.secondary(jumpStatement, null));
      }
    }
  }
}
