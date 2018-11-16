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
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;

@Rule(key = NestedControlFlowDepthCheck.CHECK_KEY)
public class NestedControlFlowDepthCheck extends PythonCheck {

  public static final String CHECK_KEY = "S134";
  private static final int DEFAULT_MAX = 4;
  private static final String MESSAGE = "Refactor this code to not nest more than %s \"if\", \"for\", \"while\", \"try\" and \"with\" statements.";

  @RuleProperty(
    key = "max",
    defaultValue = "" + DEFAULT_MAX)
  public int max = DEFAULT_MAX;

  private Deque<AstNode> depthNodes;

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(
      PythonGrammar.IF_STMT,
      PythonGrammar.FOR_STMT,
      PythonGrammar.WHILE_STMT,
      PythonGrammar.TRY_STMT,
      PythonGrammar.WITH_STMT);
  }

  @Override
  public void visitFile(AstNode astNode) {
    depthNodes = new ArrayDeque<>();
  }

  @Override
  public void visitNode(AstNode node) {
    AstNode stmtKeywordNode = node.getFirstChild();
    depthNodes.push(stmtKeywordNode);
    if (depthNodes.size() == max + 1) {
      PreciseIssue issue = addIssue(stmtKeywordNode, String.format(MESSAGE, max));

      Iterator<AstNode> depthNodesIterator = depthNodes.iterator();

      // skip current node
      depthNodesIterator.next();

      while (depthNodesIterator.hasNext()) {
        issue.secondary(depthNodesIterator.next(), "Nesting +1");
      }
    }
  }

  @Override
  public void leaveNode(AstNode astNode) {
    depthNodes.pop();
  }
}
