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
import java.util.Collections;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;

@Rule(key = ProcessSignallingCheck.CHECK_KEY)
public class ProcessSignallingCheck extends PythonCheck {
  public static final String CHECK_KEY = "S4828";
  private static final String MESSAGE = "Make sure that sending signals is safe here.";
  private AstNode scopeTree;
  private Set<String> questionableFunctions = immutableSet("kill", "killpg");

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.CALL_EXPR);
  }

  @Override
  public void visitFile(AstNode node) {
    scopeTree = node;
    super.visitFile(node);
  }

  @Override
  public void visitNode(AstNode node) {
    AstNode attributeRef = node.getFirstChild(PythonGrammar.ATTRIBUTE_REF);
    String functionName = null;
    String symbolName = null;
    if (attributeRef != null) {
      String namespace = attributeRef.getFirstChild(PythonGrammar.ATOM).getTokenValue();
      functionName = attributeRef.getFirstChild(PythonGrammar.NAME).getTokenValue();
      symbolName = namespace + "." + functionName;
    } else {
      AstNode functionNameNode = node.getFirstChild(PythonGrammar.ATOM);
      if (functionNameNode != null) {
        functionName = functionNameNode.getTokenValue();
        symbolName = functionName;
      }
    }
    if (functionName != null) {
      checkModuleName(node, functionName, symbolName);
    }
  }


  private void checkModuleName(AstNode node, String functionName, String symbolName) {
    if (questionableFunctions.contains(functionName)) {
      getContext().symbolTable().symbols(scopeTree).stream()
        .filter(symbol -> symbol.name().equals(symbolName))
        .findFirst()
        .ifPresent(symbol -> {
          if (symbol.moduleName().equals("os")) {
            addIssue(node, MESSAGE);
          }
        });
    }
  }

}
