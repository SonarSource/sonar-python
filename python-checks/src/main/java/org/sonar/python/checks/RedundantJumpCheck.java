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
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.FileInput;
import org.sonar.python.api.tree.FunctionDef;
import org.sonar.python.api.tree.ReturnStatement;
import org.sonar.python.api.tree.StatementList;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;

@Rule(key = "S3626")
public class RedundantJumpCheck extends PythonSubscriptionCheck {

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
      if (cfgBlock.successors().size() == 1 && cfgBlock.successors().contains(cfgBlock.syntacticSuccessor())) {
        List<Tree> elements = cfgBlock.elements();
        Tree lastElement = elements.get(elements.size() - 1);
        if (!isInsideSingleStatementBlock(lastElement) && !isReturnWithExpression(lastElement)) {
          ctx.addIssue(lastElement, "Remove this redundant jump.");
        }
      }
    }
  }

  // assumption: parent of BREAK, CONTINUE and RETURN is always a StatementList
  private static boolean isInsideSingleStatementBlock(Tree lastElement) {
    StatementList block = (StatementList) lastElement.parent();
    return block.statements().size() == 1;
  }

  private static boolean isReturnWithExpression(Tree lastElement) {
    return lastElement.is(Kind.RETURN_STMT) && !((ReturnStatement) lastElement).expressions().isEmpty();
  }

}
