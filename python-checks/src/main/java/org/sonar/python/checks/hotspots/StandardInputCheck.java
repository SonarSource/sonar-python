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
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.checks.AbstractCallExpressionCheck;
import org.sonar.python.semantic.Symbol;

@Rule(key = StandardInputCheck.CHECK_KEY)
public class StandardInputCheck extends AbstractCallExpressionCheck {
  public static final String CHECK_KEY = "S4829";
  private static final String MESSAGE = "Make sure that reading the standard input is safe here.";
  private static final Set<String> questionableFunctions = immutableSet("fileinput.input", "fileinput.FileInput");
  private static final Set<String> questionablePropertyAccess = immutableSet("sys.stdin", "sys.__stdin__");

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(PythonGrammar.CALL_EXPR, PythonGrammar.ATTRIBUTE_REF, PythonGrammar.ATOM);
  }

  @Override
  public void visitNode(AstNode node) {
    if (node.is(PythonGrammar.ATTRIBUTE_REF, PythonGrammar.ATOM)) {
      if (isQuestionablePropertyAccess(node)) {
        addIssue(node, message());
      }
    } else {
      if (node.getTokenValue().equals("raw_input") || node.getTokenValue().equals("input")) {
        addIssue(node, message());
      } else {
        super.visitNode(node);
      }
    }
  }

  @Override
  protected boolean isException(AstNode callExpression) {
    return callExpression.getFirstChild(PythonGrammar.ARGLIST) != null;
  }

  private boolean isQuestionablePropertyAccess(AstNode attributeRef) {
    Symbol symbol = getContext().symbolTable().getSymbol(attributeRef);
    return symbol != null && questionablePropertyAccess.contains(symbol.qualifiedName());
  }

  @Override
  protected Set<String> functionsToCheck() {
    return questionableFunctions;
  }

  @Override
  protected String message() {
    return MESSAGE;
  }
}
