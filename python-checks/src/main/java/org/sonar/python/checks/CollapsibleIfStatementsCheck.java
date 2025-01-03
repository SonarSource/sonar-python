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

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.ASSIGNMENT_EXPRESSION;

@Rule(key = CollapsibleIfStatementsCheck.CHECK_KEY)
public class CollapsibleIfStatementsCheck extends PythonVisitorCheck {
  public static final String CHECK_KEY = "S1066";
  private static final String MESSAGE = "Merge this if statement with the enclosing one.";
  private static final int MAX_LINE_LENGTH = 80;

  // consider 'and' plus 2 surrounding spaces
  private static final int AND_LENGTH = 5;

  private Set<Tree> ignored = new HashSet<>();

  @Override
  public void scanFile(PythonVisitorContext visitorContext) {
    ignored.clear();
    super.scanFile(visitorContext);
  }

  @Override
  public void visitIfStatement(IfStatement ifStatement) {
    List<Statement> statements = ifStatement.body().statements();
    if (!ifStatement.elifBranches().isEmpty()) {
      if (ifStatement.elseBranch() == null) {
        ignored.addAll(ifStatement.elifBranches().subList(0, ifStatement.elifBranches().size() - 1));
      } else {
        ignored.addAll(ifStatement.elifBranches());
      }
    }
    if (!ignored.contains(ifStatement)
      && ifStatement.elseBranch() == null
      && ifStatement.elifBranches().isEmpty()
      && statements.size() == 1
      && statements.get(0).is(Tree.Kind.IF_STMT)) {
      IfStatement singleIfChild = (IfStatement) statements.get(0);
      if (isException(singleIfChild, ifStatement)) {
        return;
      }
      addIssue(singleIfChild.keyword(), MESSAGE).secondary(ifStatement.keyword(), "enclosing");
    }
    super.visitIfStatement(ifStatement);
  }

  private static boolean isException(IfStatement singleIfChild, IfStatement enclosingIfStatement) {
    return singleIfChild.isElif()
      || singleIfChild.elseBranch() != null
      || !singleIfChild.elifBranches().isEmpty()
      || wouldCauseLongLineLength(singleIfChild, enclosingIfStatement)
      || singleIfChild.condition().is(ASSIGNMENT_EXPRESSION)
      || enclosingIfStatement.condition().is(ASSIGNMENT_EXPRESSION)
      || hasCommentsBetweenEnclosingAndChildIf(singleIfChild, enclosingIfStatement);
  }

  private static boolean hasCommentsBetweenEnclosingAndChildIf(IfStatement singleIfChild, IfStatement enclosingIfStatement) {
    return TreeUtils.tokens(enclosingIfStatement).stream()
      .anyMatch(token -> !token.trivia().isEmpty() && token.trivia().get(0).token().line() < singleIfChild.firstToken().line());
  }

  private static boolean wouldCauseLongLineLength(IfStatement singleIfChild, IfStatement enclosingIf) {
    int childConditionLength = lastColumn(singleIfChild) - singleIfChild.condition().firstToken().column();
    return (lastColumn(enclosingIf) + childConditionLength) + AND_LENGTH > MAX_LINE_LENGTH;
  }

  private static int lastColumn(IfStatement ifStatement) {
    Token lastToken = ifStatement.condition().lastToken();
    return lastToken.column() + lastToken.value().length();
  }
}
