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
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.semantic.Symbol;

@Rule(key = DebugModeCheck.CHECK_KEY)
public class DebugModeCheck extends PythonCheckAstNode {
  public static final String CHECK_KEY = "S4507";
  private static final String MESSAGE = "Make sure this debug feature is deactivated before delivering the code in production.";
  private static final Set<String> debugProperties = immutableSet("DEBUG", "DEBUG_PROPAGATE_EXCEPTIONS");

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(PythonGrammar.CALL_EXPR, PythonGrammar.EXPRESSION_STMT);
  }

  @Override
  public void visitNode(AstNode node) {
    if (node.is(PythonGrammar.EXPRESSION_STMT)) {
      if (getContext().pythonFile().fileName().equals("global_settings.py") && node.hasDirectChildren(PythonPunctuator.ASSIGN)) {
        checkDebugAssignment(node);
      }
    } else {
      AstNode attributeRef = node.getFirstChild(PythonGrammar.ATTRIBUTE_REF);
      AstNode argList = node.getFirstChild(PythonGrammar.ARGLIST);
      if (argList != null && attributeRef != null &&
        getQualifiedName(attributeRef.getFirstChild()).equals("django.conf.settings")) {
        String functionName = attributeRef.getLastChild(PythonGrammar.NAME).getTokenValue();
        if (functionName.equals("configure")) {
          argList.getChildren(PythonGrammar.ARGUMENT)
            .forEach(this::checkDebugAssignment);
        }
      }
    }
  }

  private void checkDebugAssignment(AstNode node) {
    AstNode lhsExpression = node.getFirstChild(PythonGrammar.TESTLIST_STAR_EXPR, PythonGrammar.TEST);
    AstNode rhsExpression = node.getLastChild(PythonGrammar.TESTLIST_STAR_EXPR, PythonGrammar.TEST);
    if (lhsExpression != null && rhsExpression != null
      && debugProperties.contains(lhsExpression.getTokenValue()) && rhsExpression.getTokenValue().equals("True")) {
      addIssue(node, MESSAGE);
    }
  }

  private String getQualifiedName(AstNode node) {
    Symbol symbol = getContext().symbolTable().getSymbol(node);
    return symbol != null ? symbol.qualifiedName() : "";
  }

}
