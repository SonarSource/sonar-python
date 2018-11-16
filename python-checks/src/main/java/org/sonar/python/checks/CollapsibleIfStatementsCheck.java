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
import org.sonar.sslr.ast.AstSelect;

@Rule(key = CollapsibleIfStatementsCheck.CHECK_KEY)
public class CollapsibleIfStatementsCheck extends PythonCheck {
  public static final String CHECK_KEY = "S1066";
  private static final String MESSAGE = "Merge this if statement with the enclosing one.";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.IF_STMT);
  }

  @Override
  public void visitNode(AstNode node) {
    AstNode suite = node.getLastChild(PythonGrammar.SUITE);
    if (suite.getPreviousSibling().getPreviousSibling().is(PythonKeyword.ELSE)) {
      return;
    }
    AstNode singleIfChild = singleIfChild(suite);
    if (singleIfChild != null && !hasElseOrElif(singleIfChild)) {
      addIssue(singleIfChild.getToken(), MESSAGE)
        .secondary(node.getFirstChild(), "enclosing");
    }
  }

  private static boolean hasElseOrElif(AstNode ifNode) {
    return ifNode.hasDirectChildren(PythonKeyword.ELIF) || ifNode.hasDirectChildren(PythonKeyword.ELSE);
  }

  private static AstNode singleIfChild(AstNode suite) {
    List<AstNode> statements = suite.getChildren(PythonGrammar.STATEMENT);
    if (statements.size() == 1) {
      AstSelect nestedIf = statements.get(0).select()
        .children(PythonGrammar.COMPOUND_STMT)
        .children(PythonGrammar.IF_STMT);
      if (nestedIf.size() == 1) {
        return nestedIf.get(0);
      }
    }
    return null;
  }
}
