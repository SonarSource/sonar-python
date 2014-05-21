/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python.checks;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import org.sonar.check.BelongsToProfile;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.squidbridge.checks.SquidCheck;
import org.sonar.sslr.ast.AstSelect;

import java.util.List;

@Rule(
  key = "S1066",
  priority = Priority.MAJOR)
@BelongsToProfile(title = CheckList.SONAR_WAY_PROFILE, priority = Priority.MAJOR)
public class CollapsibleIfStatementsCheck extends SquidCheck<Grammar> {

  @Override
  public void init() {
    subscribeTo(PythonGrammar.IF_STMT);
  }

  @Override
  public void visitNode(AstNode node) {
    AstNode suite = node.getLastChild(PythonGrammar.SUITE);
    if (suite.getPreviousSibling().getPreviousSibling().is(PythonKeyword.ELSE)) {
      return;
    }
    AstNode singleIfChild = singleIfChild(suite);
    if (singleIfChild != null && !hasElseOrElif(singleIfChild)) {
      getContext().createLineViolation(this, "Merge this if statement with the enclosing one.", singleIfChild);
    }
  }

  private boolean hasElseOrElif(AstNode ifNode) {
    return ifNode.hasDirectChildren(PythonKeyword.ELIF) || ifNode.hasDirectChildren(PythonKeyword.ELSE);
  }

  private AstNode singleIfChild(AstNode suite) {
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
