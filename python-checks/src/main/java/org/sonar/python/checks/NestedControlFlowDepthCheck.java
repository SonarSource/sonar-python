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
import org.sonar.check.RuleProperty;
import org.sonar.python.api.PythonGrammar;
import org.sonar.squidbridge.checks.SquidCheck;

@Rule(
  key = "S134",
  priority = Priority.MAJOR)
@BelongsToProfile(title = CheckList.SONAR_WAY_PROFILE, priority = Priority.MAJOR)
public class NestedControlFlowDepthCheck extends SquidCheck<Grammar> {

  private static final int DEFAULT_MAX = 3;

  @RuleProperty(
    key = "max",
    defaultValue = "" + DEFAULT_MAX)
  public int max = DEFAULT_MAX;

  private int depth;

  @Override
  public void init() {
    subscribeTo(
      PythonGrammar.IF_STMT,
      PythonGrammar.FOR_STMT,
      PythonGrammar.WHILE_STMT,
      PythonGrammar.TRY_STMT,
      PythonGrammar.WITH_STMT);
  }

  @Override
  public void visitFile(AstNode astNode) {
    depth = 0;
  }

  @Override
  public void visitNode(AstNode node) {
    depth++;
    if (depth == max + 1) {
      String message = "Refactor this code to not nest more than {0} \"if\", \"for\", \"while\", \"try\" and \"with\" statements.";
      getContext().createLineViolation(this, message, node, max);
    }
  }

  @Override
  public void leaveNode(AstNode astNode) {
    depth--;
  }
}

