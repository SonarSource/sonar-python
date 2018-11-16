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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.sslr.ast.AstSelect;

@Rule(key = TooManyReturnsCheck.CHECK_KEY)
public class TooManyReturnsCheck extends PythonCheck {
  public static final String CHECK_KEY = "S1142";

  private static final int DEFAULT_MAX = 3;
  private static final String MESSAGE = "This function has %s returns or yields, which is more than the %s allowed.";

  @RuleProperty(key = "max", defaultValue = "" + DEFAULT_MAX)
  public int max = DEFAULT_MAX;

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.FUNCDEF);
  }

  @Override
  public void visitNode(AstNode node) {
    AstSelect returnStatements = node.select().descendants(PythonGrammar.RETURN_STMT, PythonGrammar.YIELD_STMT);
    List<AstNode> returnNodes = new ArrayList<>();
    for (AstNode returnStatement : returnStatements){
      if (CheckUtils.insideFunction(returnStatement, node)){
        returnNodes.add(returnStatement);
      }
    }

    if (returnNodes.size() > max) {
      String message = String.format(MESSAGE, returnNodes.size(), max);
      PreciseIssue issue = addIssue(node.getFirstChild(PythonGrammar.FUNCNAME), message);
      returnNodes.forEach(returnNode -> issue.secondary(returnNode, null));
    }
  }
}

