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
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;

@Rule(key = ReturnAndYieldInOneFunctionCheck.CHECK_KEY)
public class ReturnAndYieldInOneFunctionCheck extends PythonCheck {

  public static final String MESSAGE = "Use only \"return\" or only \"yield\", not both.";

  public static final String CHECK_KEY = "S2712";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.FUNCDEF);
  }

  @Override
  public void visitNode(AstNode node) {
    List<AstNode> returnStatements = node.getDescendants(PythonGrammar.RETURN_STMT);
    List<AstNode> yieldStatements = node.getDescendants(PythonGrammar.YIELD_STMT);
    if (yieldStatements.isEmpty() || allInNestedFunction(yieldStatements, node)){
      return;
    }
    for (AstNode returnStatement : returnStatements){
      if (returnHasArgument(returnStatement) && CheckUtils.insideFunction(returnStatement, node)){
        addIssue(node.getFirstChild(PythonGrammar.FUNCNAME), MESSAGE);
        return;
      }
    }
  }

  private static boolean returnHasArgument(AstNode returnStatement) {
    return returnStatement.getFirstChild(PythonGrammar.TESTLIST) != null;
  }

  private static boolean allInNestedFunction(List<AstNode> statements, AstNode func) {
    for (AstNode statement : statements){
      if (CheckUtils.insideFunction(statement, func)){
        return false;
      }
    }
    return true;
  }
}

