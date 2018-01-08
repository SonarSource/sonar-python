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
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.semantic.Symbol;

@Rule(key = "S1481")
public class UnusedLocalVariableCheck extends PythonCheck {

  private static final String MESSAGE = "Remove the unused local variable \"%s\".";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.FUNCDEF);
  }

  @Override
  public void visitNode(AstNode functionTree) {
    // https://docs.python.org/3/library/functions.html#locals
    if (isCallingLocalsFunction(functionTree)) {
      return;
    }
    for (Symbol symbol : getContext().symbolTable().symbols(functionTree)) {
      checkSymbol(symbol);
    }
  }

  private static boolean isCallingLocalsFunction(AstNode functionTree) {
    return functionTree
      .getDescendants(PythonGrammar.NAME)
      .stream()
      .anyMatch(node -> "locals".equals(node.getTokenValue()));
  }

  private void checkSymbol(Symbol symbol) {
    if (symbol.readUsages().isEmpty()) {
      for (AstNode writeUsage : symbol.writeUsages()) {
        if (!writeUsage.hasAncestor(PythonGrammar.TYPEDARGSLIST)) {
          addIssue(writeUsage, String.format(MESSAGE, symbol.name()));
        }
      }
    }
  }
}
