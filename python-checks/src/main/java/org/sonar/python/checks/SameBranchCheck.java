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

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1871")
public class SameBranchCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Either merge this branch with the identical one on line \"%s\" or change one of the implementations.";

  private List<Tree> ignoreList;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> ignoreList = new ArrayList<>());

    context.registerSyntaxNodeConsumer(Tree.Kind.IF_STMT, ctx -> {
      IfStatement ifStmt = (IfStatement) ctx.syntaxNode();
      if (ignoreList.contains(ifStmt)) {
        return;
      }
      boolean hasElseClause = ifStmt.elseBranch() != null;
      //In this case, S3923 will raise a bug
      if (hasElseClause && allIdenticalBranches(ifStmt)) {
        return;
      }
      List<StatementList> branches = getIfBranches(ifStmt);
      findSameBranches(branches, ctx);
    });
  }

  private static boolean allIdenticalBranches(IfStatement ifStmt) {
    StatementList body = ifStmt.body();
    for (IfStatement elifBranch : ifStmt.elifBranches()) {
      if (!CheckUtils.areEquivalent(body, elifBranch.body())) {
        return false;
      }
    }
    return CheckUtils.areEquivalent(body, ifStmt.elseBranch().body());
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
        if (!isOnASingleLine && !allBranchesIdentical) {
          int line = TreeUtils.nonWhitespaceTokens(originalBlock).get(0).line();
          String message = String.format(MESSAGE, line);
          List<Token> issueTokens = TreeUtils.nonWhitespaceTokens(duplicateBlock);
          ctx.addIssue(issueTokens.get(0), issueTokens.get(issueTokens.size() - 1), message)
            .secondary(issueLocation(originalBlock, "Original"));
          break;
        }
        if (allBranchesIdentical) {
          equivalentBlocks.add(duplicateBlock);
          Tree firstBlock = branches.get(0);
          int line = TreeUtils.nonWhitespaceTokens(firstBlock).get(0).line();
          String message = String.format(MESSAGE, line);
          equivalentBlocks.stream().skip(1).forEach(e -> {
            List<Token> issueTokens = TreeUtils.nonWhitespaceTokens(e);
            ctx.addIssue(issueTokens.get(0), issueTokens.get(issueTokens.size() - 1), message)
              .secondary(issueLocation(firstBlock, "Original"));
          });
        }
      }
    }
  }

  private static IssueLocation issueLocation(Tree body, String message) {
    List<Token> tokens = TreeUtils.nonWhitespaceTokens(body);
    return IssueLocation.preciseLocation(tokens.get(0), tokens.get(tokens.size() - 1), message);
  }

  private List<StatementList> getIfBranches(IfStatement ifStmt) {
    List<StatementList> branches = new ArrayList<>();
    branches.add(ifStmt.body());
    branches.addAll(ifStmt.elifBranches().stream().map(IfStatement::body).toList());
    ElseClause elseClause = ifStmt.elseBranch();
    if (elseClause != null) {
      branches.add(elseClause.body());
      lookForElseIfs(branches, elseClause);
    }
    return branches;
  }

  private void lookForElseIfs(List<StatementList> branches, ElseClause elseBranch) {
    IfStatement singleIfChild = singleIfChild(elseBranch.body());
    if (singleIfChild != null) {
      ignoreList.add(singleIfChild);
      branches.addAll(getIfBranches(singleIfChild));
    }
  }

  private static IfStatement singleIfChild(StatementList statementList) {
    List<Statement> statements = statementList.statements();
    if (statements.size() == 1 && statements.get(0).is(Tree.Kind.IF_STMT)) {
      return (IfStatement) statements.get(0);
    }
    return null;
  }

  private static boolean isOnASingleLine(Tree tree) {
    if (tree.is(Tree.Kind.STATEMENT_LIST)) {
      StatementList duplicateBlock = (StatementList) tree;
      return duplicateBlock.statements().get(0).firstToken().line() == duplicateBlock.statements().get(duplicateBlock.statements().size() - 1).lastToken().line();
    } else {
      return tree.firstToken().line() == tree.lastToken().line();
    }
  }
}
