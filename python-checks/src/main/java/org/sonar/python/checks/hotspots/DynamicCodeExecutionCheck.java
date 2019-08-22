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
package org.sonar.python.checks.hotspots;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheckAstNode;
import org.sonar.python.api.PythonGrammar;

@Rule(key = DynamicCodeExecutionCheck.CHECK_KEY)
public class DynamicCodeExecutionCheck extends PythonCheckAstNode {
  public static final String CHECK_KEY = "S1523";
  private static final String MESSAGE = "Make sure that this dynamic injection or execution of code is safe.";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(PythonGrammar.CALL_EXPR);
  }

  @Override
  public void visitNode(AstNode callExpression) {
    AstNode functionNameNode = callExpression.getFirstChild(PythonGrammar.ATOM);
    if (functionNameNode != null) {
      String functionName = functionNameNode.getTokenValue();
      if (functionName.equals("exec") || functionName.equals("eval")) {
        addIssue(callExpression, MESSAGE);
      }
    }
  }
}
