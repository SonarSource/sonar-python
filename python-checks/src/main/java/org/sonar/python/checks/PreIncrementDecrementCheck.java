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
import org.sonar.python.api.PythonPunctuator;
import org.sonar.squidbridge.checks.SquidCheck;

import java.util.List;

@Rule(
  key = "PreIncrementDecrement",
  priority = Priority.MAJOR)
@BelongsToProfile(title = CheckList.SONAR_WAY_PROFILE, priority = Priority.MAJOR)
public class PreIncrementDecrementCheck extends SquidCheck<Grammar> {

  @Override
  public void init() {
    subscribeTo(PythonGrammar.FACTOR);
  }

  @Override
  public void visitNode(AstNode astNode) {
    List<AstNode> children = astNode.getChildren();
    AstNode firstChild = children.get(0);
    AstNode secondChild = children.get(1);
    if (firstChild.is(PythonPunctuator.PLUS) && secondChild.getFirstChild().is(PythonPunctuator.PLUS)) {
      getContext().createLineViolation(this, "This statement doesn't produce the expected result, replace use of non-existent pre-increment operator", astNode);
    }
    if (firstChild.is(PythonPunctuator.MINUS) && secondChild.getFirstChild().is(PythonPunctuator.MINUS)) {
      getContext().createLineViolation(this, "This statement doesn't produce the expected result, replace use of non-existent pre-decrement operator", astNode);
    }
  }

}
