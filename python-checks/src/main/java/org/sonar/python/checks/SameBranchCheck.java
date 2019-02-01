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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.python.IssueLocation;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;
import org.sonar.sslr.ast.AstSelect;

@Rule(key = SameBranchCheck.CHECK_KEY)
public class SameBranchCheck extends PythonCheck {
  public static final String CHECK_KEY = "S1871";
  public static final String MESSAGE = "Either merge this branch with the identical one on line \"%s\" or change one of the implementations.";

  private static final int CONDITIONAL_EXPRESSION_SIZE = 5;
  private static final int CONDITIONAL_EXPRESSION_TRUE_BRANCH = 0;
  private static final int CONDITIONAL_EXPRESSION_IF = 1;
  private static final int CONDITIONAL_EXPRESSION_FALSE_BRANCH = 4;

  private List<AstNode> ignoreList;

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(PythonGrammar.IF_STMT, PythonGrammar.TEST);
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
    List<AstNode> branches = node.is(PythonGrammar.IF_STMT) ? getIfBranches(node) : getConditionalExpressionBranches(node);
    findSameBranches(branches);
  }

  private List<AstNode> getConditionalExpressionBranches(AstNode node) {
    List<AstNode> branches = new ArrayList<>();
    appendConditionalExpressionBranches(branches, node);
    return branches;
  }

  private void appendConditionalExpressionBranches(List<AstNode> branches, AstNode node) {
    if (node.is(PythonGrammar.TEST)) {
      ignoreList.add(node);
      List<AstNode> children = node.getChildren();
      if (children.size() == 1) {
        appendConditionalExpressionBranches(branches, children.get(0));
      } else if (children.size() == CONDITIONAL_EXPRESSION_SIZE && children.get(CONDITIONAL_EXPRESSION_IF).is(PythonKeyword.IF)) {
        appendConditionalExpressionBranches(branches, children.get(CONDITIONAL_EXPRESSION_TRUE_BRANCH));
        appendConditionalExpressionBranches(branches, children.get(CONDITIONAL_EXPRESSION_FALSE_BRANCH));
      }
    } else if (node.is(PythonGrammar.ATOM) && node.getNumberOfChildren() == 3 && node.getFirstChild().is(PythonPunctuator.LPARENTHESIS)) {
      appendConditionalExpressionBranches(branches, node.getChildren().get(1));
    } else if (node.is(PythonGrammar.TESTLIST_COMP) && node.getNumberOfChildren() == 1) {
      appendConditionalExpressionBranches(branches, node.getFirstChild());
    } else {
      branches.add(node);
    }
  }

  private List<AstNode> getIfBranches(AstNode ifStmt) {
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
      branches.addAll(getIfBranches(singleIfChild));
    }
  }

  private void findSameBranches(List<AstNode> branches) {
    for (int i = 1; i < branches.size(); i++) {
      checkBranch(branches, i);
    }
  }

  private void checkBranch(List<AstNode> branches, int index) {
    AstNode duplicateBlock = branches.get(index);
    boolean isOnASingleLine = isOnASingleLine(duplicateBlock);
    List<AstNode> equivalentBlocks = new ArrayList<>();
    for (int j = 0; j < index; j++) {
      AstNode originalBlock = branches.get(j);
      if (CheckUtils.equalNodes(originalBlock, duplicateBlock)) {
        equivalentBlocks.add(originalBlock);
        boolean isLastComparisonInBranches = j == branches.size() - 2;
        if (!isOnASingleLine || isLastComparisonInBranches) {
          int line = originalBlock.getTokenLine() + (originalBlock.is(PythonGrammar.SUITE) ? 1 : 0);
          String message = String.format(MESSAGE, line);
          PreciseIssue issue = addIssue(location(duplicateBlock, message));
          equivalentBlocks.forEach(original -> issue.secondary(location(original, "Original")));
          return;
        }
      } else if (isOnASingleLine) {
        return;
      }
    }
  }

  private static IssueLocation location(AstNode node, String message) {
    AstNode firstStatement = node.getFirstChild(PythonGrammar.STATEMENT);
    if (firstStatement != null) {
      return IssueLocation.preciseLocation(firstStatement, getLastNode(node), message);
    } else {
      return IssueLocation.preciseLocation(node, message);
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

  public static boolean isOnASingleLine(AstNode parent) {
    List<AstNode> statements = parent.getChildren(PythonGrammar.STATEMENT);
    if (statements.isEmpty()) {
      return true;
    }
    return statements.get(0).getTokenLine() == statements.get(statements.size() - 1).getLastToken().getLine();
  }
}
