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
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;

/**
 * Note that implementation differs from AbstractOneStatementPerLineCheck due to Python specifics
 */
@Rule(key = OneStatementPerLineCheck.CHECK_KEY)
public class OneStatementPerLineCheck extends PythonCheck {

  public static final String CHECK_KEY = "OneStatementPerLine";
  private final Map<Integer, Integer> statementsPerLine = new HashMap<>();

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(PythonGrammar.SIMPLE_STMT, PythonGrammar.SUITE);
  }

  @Override
  public void visitFile(AstNode astNode) {
    statementsPerLine.clear();
  }

  @Override
  public void visitNode(AstNode statementNode) {
    int line = statementNode.getTokenLine();
    if (!statementsPerLine.containsKey(line)) {
      statementsPerLine.put(line, 0);
    }
    statementsPerLine.put(line, statementsPerLine.get(line) + 1);
  }

  @Override
  public void leaveFile(AstNode astNode) {
    for (Map.Entry<Integer, Integer> statementsAtLine : statementsPerLine.entrySet()) {
      if (statementsAtLine.getValue() > 1) {
        String message = String.format("At most one statement is allowed per line, but %s statements were found on this line.", statementsAtLine.getValue());
        int lineNumber = statementsAtLine.getKey();
        addLineIssue(message, lineNumber);
      }
    }
  }

}
