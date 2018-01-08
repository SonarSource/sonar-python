/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.python.IssueLocation;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonTokenType;
import org.sonar.sslr.ast.AstSelect;

import static org.sonar.python.api.PythonGrammar.STMT_LIST;

@Rule(key = SameBranchCheck.CHECK_KEY)
public class SameBranchCheck extends PythonCheck {
  public static final String CHECK_KEY = "S1871";
  public static final String MESSAGE = "Either merge this branch with the identical one on line \"%s\" or change one of the implementations.";

  private List<AstNode> ignoreList;

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.IF_STMT);
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
    AstNode duplicateBlock = branches.get(index);
    for (int j = 0; j < index; j++) {
      AstNode originalBlock = branches.get(j);
      if (CheckUtils.equalNodes(originalBlock, duplicateBlock)) {
        String message = String.format(MESSAGE, originalBlock.getToken().getLine() + 1);
        addIssue(location(duplicateBlock, message))
          .secondary(location(originalBlock, "Original"));
        return;
      }
    }
  }

  private static IssueLocation location(AstNode suiteNode, String message) {
    AstNode firstStatement = suiteNode.getFirstChild(PythonGrammar.STATEMENT);
    if (firstStatement != null) {
      return IssueLocation.preciseLocation(firstStatement, getLastNode(suiteNode), message);
    } else {
      return IssueLocation.preciseLocation(suiteNode.getFirstChild(STMT_LIST), message);
    }
  }

  /**
   * We need this method to avoid passing of new line or dedent as pointer to end of issue location
   */
  private static AstNode getLastNode(AstNode node) {
    if (node.getNumberOfChildren() == 0) {
      return node;
    }

    AstNode lastChild = node.getLastChild();
    while (lastChild.is(PythonTokenType.NEWLINE, PythonTokenType.DEDENT, PythonTokenType.INDENT)) {
      lastChild = lastChild.getPreviousSibling();
    }

    return getLastNode(lastChild);
  }

  private static AstNode singleIfChild(AstNode suite) {
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
