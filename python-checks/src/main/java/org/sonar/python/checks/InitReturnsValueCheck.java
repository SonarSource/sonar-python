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
import org.sonar.python.api.PythonKeyword;

@Rule(key = InitReturnsValueCheck.CHECK_KEY)
public class InitReturnsValueCheck extends PythonCheck {

  public static final String MESSAGE_RETURN = "Remove this return value.";
  public static final String MESSAGE_YIELD = "Remove this yield statement.";

  public static final String CHECK_KEY = "S2734";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.FUNCDEF);
  }

  @Override
  public void visitNode(AstNode node) {
    if (!"__init__".equals(node.getFirstChild(PythonGrammar.FUNCNAME).getTokenValue())){
      return;
    }
    List<AstNode> returnYieldStatements = node.getDescendants(PythonGrammar.YIELD_STMT, PythonGrammar.RETURN_STMT);
    for (AstNode returnYieldStatement : returnYieldStatements){
      if (CheckUtils.insideFunction(returnYieldStatement, node) && !returnReturnNone(returnYieldStatement)){
        raiseIssue(returnYieldStatement);
      }
    }
  }

  private static boolean returnReturnNone(AstNode stmt) {
    return stmt.is(PythonGrammar.RETURN_STMT)
        && (stmt.getFirstChild(PythonGrammar.TESTLIST) == null
        || stmt.getFirstChild(PythonGrammar.TESTLIST).getToken().getValue().equals(PythonKeyword.NONE.getValue()));
  }

  private void raiseIssue(AstNode node) {
    String message = MESSAGE_RETURN;
    if (node.is(PythonGrammar.YIELD_STMT)){
      message = MESSAGE_YIELD;
    }
    addIssue(node, message);
  }
}

