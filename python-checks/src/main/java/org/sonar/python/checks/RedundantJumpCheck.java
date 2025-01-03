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

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.cfg.PythonCfgBranchingBlock;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S3626")
public class RedundantJumpCheck extends PythonSubscriptionCheck {

  public static final String QUICK_FIX_DESCRIPTION = "Remove redundant statement";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, ctx -> checkCfg(ControlFlowGraph.build((FileInput) ctx.syntaxNode(), ctx.pythonFile()), ctx));
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx -> checkCfg(ControlFlowGraph.build((FunctionDef) ctx.syntaxNode(), ctx.pythonFile()), ctx));
  }

  private static void checkCfg(@Nullable ControlFlowGraph cfg, SubscriptionContext ctx) {
    Optional.ofNullable(cfg)
      .map(ControlFlowGraph::blocks)
      .stream()
      .flatMap(Collection::stream)
      .filter(cfgBlock -> cfgBlock.successors().size() == 1 && cfgBlock.successors().contains(cfgBlock.syntacticSuccessor()))
      .forEach(cfgBlock -> {
        List<Tree> elements = cfgBlock.elements();
        Tree lastElement = elements.get(elements.size() - 1);
        if (!isException(lastElement)) {
          var issue = ctx.addIssue(lastElement, message(lastElement));
          addQuickFix(lastElement, issue);

          if (lastElement.is(Kind.CONTINUE_STMT)) {
            Tree loop = ((PythonCfgBranchingBlock) cfgBlock.successors().iterator().next()).branchingTree();
            issue.secondary(loop.firstToken(), null);
          }
        }
      });
  }

  private static void addQuickFix(Tree lastElement, PreciseIssue issue) {
    if (!(lastElement instanceof Statement statement)) {
      return;
    }
    var quickFix = PythonQuickFix
      .newQuickFix(QUICK_FIX_DESCRIPTION)
      .addTextEdit(TextEditUtils.removeStatement(statement))
      .build();
    issue.addQuickFix(quickFix);
  }

  private static String message(Tree jumpStatement) {
    String jumpKind = jumpStatement.is(Kind.RETURN_STMT) ? "return" : "continue";
    return "Remove this redundant " + jumpKind + ".";
  }

  // assumption: parent of BREAK, CONTINUE and RETURN is always a StatementList
  private static boolean isInsideSingleStatementBlock(Tree lastElement) {
    StatementList block = (StatementList) lastElement.parent();
    return block.statements().size() == 1;
  }

  private static boolean isReturnWithExpression(Tree lastElement) {
    return lastElement.is(Kind.RETURN_STMT) && !((ReturnStatement) lastElement).expressions().isEmpty();
  }

  private static boolean isException(Tree lastElement) {
    return lastElement.is(Kind.RAISE_STMT)
      || isReturnWithExpression(lastElement)
      || isInsideSingleStatementBlock(lastElement)
      || hasTryAncestor(lastElement);
  }

  // ignore jumps in try statement because CFG is not precise
  private static boolean hasTryAncestor(Tree element) {
    return TreeUtils.firstAncestorOfKind(element, Kind.TRY_STMT) != null;
  }

}
