/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;
import org.sonar.sslr.ast.AstSelect;

import javax.annotation.Nullable;
import java.util.LinkedList;
import java.util.List;

@Rule(
    key = SameBranchCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Two branches in the same conditional structure should not have exactly the same implementation",
    tags = {Tags.BUG}
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.LOGIC_RELIABILITY)
@SqaleConstantRemediation("10min")
@ActivatedByDefault
public class SameBranchCheck extends SquidCheck<Grammar> {
  public static final String CHECK_KEY = "S1871";
  public static final String MESSAGE = "Either merge this branch with the identical one on line \"%s\" or change one of the implementations.";

  private List<AstNode> ignoreList;

  @Override
  public void init() {
    subscribeTo(PythonGrammar.IF_STMT);
  }

  @Override
  public void visitFile(@Nullable AstNode astNode) {
    ignoreList = new LinkedList<>();
  }

  @Override
  public void visitNode(AstNode node) {
    if (ignoreList.contains(node)) {
      return;
    }
    List<AstNode> branches = getBranchesToCompare(node);
    findSameBranches(branches);
  }

  private List<AstNode> getBranchesToCompare(AstNode ifStmt) {
    List<AstNode> branches = ifStmt.getChildren(PythonGrammar.SUITE);
    AstNode elseNode = ifStmt.getFirstChild(PythonKeyword.ELSE);
    if (branches.size() == 2 && elseNode != null) {
      AstNode suite = branches.get(1);
      lookForElseIfs(branches, suite);
    }
    return branches;
  }

  private void lookForElseIfs(List<AstNode> branches, AstNode suite) {
    AstNode singleIfChild = singleIfChild(suite);
    if (singleIfChild != null) {
      ignoreList.add(singleIfChild);
      branches.addAll(getBranchesToCompare(singleIfChild));
    }
  }

  private void findSameBranches(List<AstNode> branches) {
    for (int i = 1; i < branches.size(); i++) {
      checkBranch(branches, i);
    }
  }

  private void checkBranch(List<AstNode> branches, int index) {
    for (int j = 0; j < index; j++) {
      if (CheckUtils.equalNodes(branches.get(j), branches.get(index))) {
        String message = String.format(MESSAGE, branches.get(j).getToken().getLine() + 1);
        getContext().createLineViolation(this, message, branches.get(index).getToken().getLine() + 1);
        return;
      }
    }
  }

  private AstNode singleIfChild(AstNode suite) {
    List<AstNode> statements = suite.getChildren(PythonGrammar.STATEMENT);
    if (statements.size() == 1) {
      AstSelect nestedIf = statements.get(0).select()
          .children(PythonGrammar.COMPOUND_STMT)
          .children(PythonGrammar.IF_STMT);
      if (nestedIf.size() == 1) {
        return nestedIf.get(0);
      }
    }
    return null;
  }
}
