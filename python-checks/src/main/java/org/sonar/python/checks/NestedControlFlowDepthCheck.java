/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.api.PythonKeyword;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.plugins.python.api.tree.WithStatement;

@Rule(key = "S134")
public class NestedControlFlowDepthCheck extends PythonVisitorCheck {

  private static final int DEFAULT_MAX = 4;
  private static final String MESSAGE = "Refactor this code to not nest more than %s \"if\", \"for\", \"while\", \"try\" and \"with\" statements.";

  @RuleProperty(
    key = "max",
    description = "Maximum allowed \"if\", \"for\", \"while\", \"try\" and \"with\" statements nesting depth",
    defaultValue = "" + DEFAULT_MAX)
  public int max = DEFAULT_MAX;

  private Deque<Token> depthNodes = new ArrayDeque<>();

  @Override
  public void scanFile(PythonVisitorContext visitorContext) {
    depthNodes.clear();
    super.scanFile(visitorContext);
  }

  @Override
  public void visitIfStatement(IfStatement pyIfStatementTree) {
    Token keyword = pyIfStatementTree.keyword();
    boolean isIFKeyword = keyword.type().equals(PythonKeyword.IF);
    if (isIFKeyword) {
      depthNodes.push(keyword);
      checkNode();
    }
    super.visitIfStatement(pyIfStatementTree);
    if (isIFKeyword) {
      depthNodes.pop();
    }
  }

  @Override
  public void visitForStatement(ForStatement pyForStatementTree) {
    depthNodes.push(pyForStatementTree.forKeyword());
    checkNode();
    super.visitForStatement(pyForStatementTree);
    depthNodes.pop();
  }

  @Override
  public void visitWhileStatement(WhileStatement pyWhileStatementTree) {
    depthNodes.push(pyWhileStatementTree.whileKeyword());
    checkNode();
    super.visitWhileStatement(pyWhileStatementTree);
    depthNodes.pop();
  }

  @Override
  public void visitTryStatement(TryStatement pyTryStatementTree) {
    depthNodes.push(pyTryStatementTree.tryKeyword());
    checkNode();
    super.visitTryStatement(pyTryStatementTree);
    depthNodes.pop();
  }

  @Override
  public void visitWithStatement(WithStatement pyWithStatementTree) {
    depthNodes.push(pyWithStatementTree.withKeyword());
    checkNode();
    super.visitWithStatement(pyWithStatementTree);
    depthNodes.pop();
  }

  private void checkNode() {
    if (depthNodes.size() == max + 1) {
      Token lastToken = depthNodes.peek();
      PreciseIssue issue = addIssue(lastToken, String.format(MESSAGE, max));

      Iterator<Token> depthNodesIterator = depthNodes.iterator();

      // skip current node
      depthNodesIterator.next();

      while (depthNodesIterator.hasNext()) {
        issue.secondary(depthNodesIterator.next(), "Nesting +1");
      }
    }
  }
}
