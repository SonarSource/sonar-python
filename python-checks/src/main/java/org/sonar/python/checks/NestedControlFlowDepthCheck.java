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

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.PythonCheckTree;
import org.sonar.python.PythonVisitorContext;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.tree.PyForStatementTree;
import org.sonar.python.api.tree.PyIfStatementTree;
import org.sonar.python.api.tree.PyToken;
import org.sonar.python.api.tree.PyTryStatementTree;
import org.sonar.python.api.tree.PyWhileStatementTree;
import org.sonar.python.api.tree.PyWithStatementTree;

@Rule(key = "S134")
public class NestedControlFlowDepthCheck extends PythonCheckTree {

  private static final int DEFAULT_MAX = 4;
  private static final String MESSAGE = "Refactor this code to not nest more than %s \"if\", \"for\", \"while\", \"try\" and \"with\" statements.";

  @RuleProperty(
    key = "max",
    defaultValue = "" + DEFAULT_MAX)
  public int max = DEFAULT_MAX;

  private Deque<PyToken> depthNodes = new ArrayDeque<>();

  @Override
  public void scanFile(PythonVisitorContext visitorContext) {
    depthNodes.clear();
    super.scanFile(visitorContext);
  }

  @Override
  public void visitIfStatement(PyIfStatementTree pyIfStatementTree) {
    PyToken keyword = pyIfStatementTree.keyword();
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
  public void visitForStatement(PyForStatementTree pyForStatementTree) {
    depthNodes.push(pyForStatementTree.forKeyword());
    checkNode();
    super.visitForStatement(pyForStatementTree);
    depthNodes.pop();
  }

  @Override
  public void visitWhileStatement(PyWhileStatementTree pyWhileStatementTree) {
    depthNodes.push(pyWhileStatementTree.whileKeyword());
    checkNode();
    super.visitWhileStatement(pyWhileStatementTree);
    depthNodes.pop();
  }

  @Override
  public void visitTryStatement(PyTryStatementTree pyTryStatementTree) {
    depthNodes.push(pyTryStatementTree.tryKeyword());
    checkNode();
    super.visitTryStatement(pyTryStatementTree);
    depthNodes.pop();
  }

  @Override
  public void visitWithStatement(PyWithStatementTree pyWithStatementTree) {
    depthNodes.push(pyWithStatementTree.firstToken());
    checkNode();
    super.visitWithStatement(pyWithStatementTree);
    depthNodes.pop();
  }

  private void checkNode() {
    if (depthNodes.size() == max + 1) {
      PyToken lastToken = depthNodes.peek();
      PreciseIssue issue = addIssue(lastToken, String.format(MESSAGE, max));

      Iterator<PyToken> depthNodesIterator = depthNodes.iterator();

      // skip current node
      depthNodesIterator.next();

      while (depthNodesIterator.hasNext()) {
        issue.secondary(depthNodesIterator.next(), "Nesting +1");
      }
    }
  }
}
