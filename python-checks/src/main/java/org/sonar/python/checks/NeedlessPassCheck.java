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
import org.sonar.python.api.PythonTokenType;
import org.sonar.sslr.ast.AstSelect;

@Rule(key = NeedlessPassCheck.CHECK_KEY)
public class NeedlessPassCheck extends PythonCheck {

  public static final String CHECK_KEY = "S2772";

  private static final String MESSAGE = "Remove this unneeded \"pass\".";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.PASS_STMT);
  }

  @Override
  public void visitNode(AstNode node) {
    AstNode suite = node.getFirstAncestor(PythonGrammar.SUITE);
    if (suite != null) {
      List<AstNode> statements = suite.getChildren(PythonGrammar.STATEMENT);
      if (statements.size() > 1) {
        if (!docstringException(statements)) {
          addIssue(node, MESSAGE);
        }
      } else {
        visitOneOrZeroStatement(node, suite, statements.size());
      }
    }
  }

  private static boolean docstringException(List<AstNode> statements) {
    return statements.size() == 2 && statements.get(0).getToken().getType().equals(PythonTokenType.STRING);
  }

  private void visitOneOrZeroStatement(AstNode node, AstNode suite, int statementNumber) {
    AstSelect simpleStatements;
    if (statementNumber == 1) {
      simpleStatements = suite.select()
          .children(PythonGrammar.STATEMENT)
          .children(PythonGrammar.STMT_LIST)
          .children(PythonGrammar.SIMPLE_STMT);
    } else {
      simpleStatements = suite.select()
          .children(PythonGrammar.STMT_LIST)
          .children(PythonGrammar.SIMPLE_STMT);
    }
    if (simpleStatements.size() > 1) {
      addIssue(node, MESSAGE);
    }
  }

}

