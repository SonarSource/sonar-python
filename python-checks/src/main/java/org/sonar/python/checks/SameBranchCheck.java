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

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.PyConditionalExpressionTree;
import org.sonar.python.api.tree.PyElseStatementTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyIfStatementTree;
import org.sonar.python.api.tree.PyParenthesizedExpressionTree;
import org.sonar.python.api.tree.PyStatementListTree;
import org.sonar.python.api.tree.PyStatementTree;
import org.sonar.python.api.tree.PyToken;
import org.sonar.python.api.tree.Tree;

@Rule(key = "S1871")
public class SameBranchCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Either merge this branch with the identical one on line \"%s\" or change one of the implementations.";

  private List<Tree> ignoreList;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> ignoreList = new ArrayList<>());

    context.registerSyntaxNodeConsumer(Tree.Kind.IF_STMT, ctx -> {
      PyIfStatementTree ifStmt = (PyIfStatementTree) ctx.syntaxNode();
      if (ignoreList.contains(ifStmt)) {
        return;
      }
      List<PyStatementListTree> branches = getIfBranches(ifStmt);
      findSameBranches(branches, ctx);
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.CONDITIONAL_EXPR, ctx -> {
      PyConditionalExpressionTree conditionalExpression = (PyConditionalExpressionTree) ctx.syntaxNode();
      if (ignoreList.contains(conditionalExpression)) {
        return;
      }
      List<PyExpressionTree> expressions = new ArrayList<>();
      addConditionalExpressionBranches(expressions, conditionalExpression);
      findSameBranches(expressions, ctx);
    });
  }

  private static void findSameBranches(List<? extends Tree> branches, SubscriptionContext ctx) {
    for (int i = 1; i < branches.size(); i++) {
      checkBranches(branches, i, ctx);
    }
  }

  private static void checkBranches(List<? extends Tree> branches, int index, SubscriptionContext ctx) {
    Tree duplicateBlock = branches.get(index);
    boolean isOnASingleLine = isOnASingleLine(duplicateBlock);
    List<Tree> equivalentBlocks = new ArrayList<>();
    for (int j = 0; j < index; j++) {
      Tree originalBlock = branches.get(j);
      if (CheckUtils.areEquivalent(originalBlock, duplicateBlock)) {
        equivalentBlocks.add(originalBlock);
        boolean allBranchesIdentical = equivalentBlocks.size() == branches.size() - 1;
        if (!isOnASingleLine || allBranchesIdentical) {
          int line = getFirstToken(originalBlock).line();
          String message = String.format(MESSAGE, line);
          PreciseIssue issue = ctx.addIssue(getFirstToken(duplicateBlock), getLastToken(duplicateBlock), message);
          equivalentBlocks.forEach(e -> issue.secondary(getFirstToken(e), "Original"));
        }
      }
    }
  }

  private List<PyStatementListTree> getIfBranches(PyIfStatementTree ifStmt) {
    List<PyStatementListTree> branches = new ArrayList<>();
    branches.add(ifStmt.body());
    branches.addAll(ifStmt.elifBranches().stream().map(PyIfStatementTree::body).collect(Collectors.toList()));
    PyElseStatementTree elseStatement = ifStmt.elseBranch();
    if (elseStatement != null) {
      branches.add(elseStatement.body());
      lookForElseIfs(branches, elseStatement);
    }
    return branches;
  }

  private void addConditionalExpressionBranches(List<PyExpressionTree> branches, PyConditionalExpressionTree conditionalExpression) {
    PyExpressionTree trueExpression = removeParentheses(conditionalExpression.trueExpression());
    PyExpressionTree falseExpression = removeParentheses(conditionalExpression.falseExpression());
    if (trueExpression.is(Tree.Kind.CONDITIONAL_EXPR)) {
      ignoreList.add(trueExpression);
      addConditionalExpressionBranches(branches, (PyConditionalExpressionTree) trueExpression);
    } else {
      branches.add(trueExpression);
    }
    if (falseExpression.is(Tree.Kind.CONDITIONAL_EXPR)) {
      ignoreList.add(falseExpression);
      addConditionalExpressionBranches(branches, (PyConditionalExpressionTree) falseExpression);
    } else {
      branches.add(falseExpression);
    }
  }

  private static PyExpressionTree removeParentheses(PyExpressionTree expression) {
    if (expression.is(Tree.Kind.PARENTHESIZED)) {
      return removeParentheses(((PyParenthesizedExpressionTree) expression).expression());
    } else {
      return expression;
    }
  }

  private void lookForElseIfs(List<PyStatementListTree> branches, PyElseStatementTree elseBranch) {
    PyIfStatementTree singleIfChild = singleIfChild(elseBranch.body());
    if (singleIfChild != null) {
      ignoreList.add(singleIfChild);
      branches.addAll(getIfBranches(singleIfChild));
    }
  }

  private static PyIfStatementTree singleIfChild(PyStatementListTree statementList) {
    List<PyStatementTree> statements = statementList.statements();
    if (statements.size() == 1 && statements.get(0).is(Tree.Kind.IF_STMT)) {
      return (PyIfStatementTree) statements.get(0);
    }
    return null;
  }

  private static boolean isOnASingleLine(Tree tree) {
    if (tree.is(Tree.Kind.STATEMENT_LIST)) {
      PyStatementListTree duplicateBlock = (PyStatementListTree) tree;
      return duplicateBlock.statements().get(0).firstToken().line() == duplicateBlock.statements().get(duplicateBlock.statements().size() - 1).lastToken().line();
    } else {
      return tree.firstToken().line() == tree.lastToken().line();
    }
  }

  private static PyToken getFirstToken(Tree tree) {
    if (tree.is(Tree.Kind.STATEMENT_LIST)) {
      return getFirstToken(((PyStatementListTree) tree).statements().get(0));
    }
    return tree.firstToken();
  }

  private static PyToken getLastToken(Tree tree) {
    if (tree.is(Tree.Kind.STATEMENT_LIST)) {
      List<PyStatementTree> statements = ((PyStatementListTree) tree).statements();
      return getLastToken(statements.get(statements.size()-1));
    }
    return tree.lastToken();
  }
}
