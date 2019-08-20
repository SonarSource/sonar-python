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
package org.sonar.python.checks;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.Collections;
import java.util.Set;
import org.sonar.python.PythonCheckAstNode;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.semantic.Symbol;

public abstract class AbstractCallExpressionCheck extends PythonCheckAstNode {

  protected abstract Set<String> functionsToCheck();

  protected abstract String message();

  protected boolean isException(AstNode callExpression) {
    return false;
  }

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.CALL_EXPR);
  }

  @Override
  public void visitNode(AstNode node) {
    Symbol symbol = getContext().symbolTable().getSymbol(node);
    if (!isException(node) && symbol != null && functionsToCheck().contains(symbol.qualifiedName())) {
      addIssue(node, message());
    }
  }
}
