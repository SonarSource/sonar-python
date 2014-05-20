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
  key = "S107",
  priority = Priority.MAJOR)
@BelongsToProfile(title = CheckList.SONAR_WAY_PROFILE, priority = Priority.MAJOR)
public class TooManyParametersCheck extends SquidCheck<Grammar> {

  private static final int DEFAULT_MAX = 7;

  @RuleProperty(
    key = "max",
    defaultValue = "" + DEFAULT_MAX)
  public int max = DEFAULT_MAX;

  @Override
  public void init() {
    subscribeTo(PythonGrammar.FUNCDEF, PythonGrammar.LAMBDEF);
  }

  @Override
  public void visitNode(AstNode node) {
    int nbParameters = node.select()
      .children(PythonGrammar.VARARGSLIST)
      .children(PythonGrammar.FPDEF)
      .size();
    if (nbParameters > max) {
      String name = "Lambda";
      if (node.is(PythonGrammar.FUNCDEF)) {
        String typeName = CheckUtils.isMethodDefinition(node) ? "Method" : "Function";
        name = node.getFirstChild(PythonGrammar.FUNCNAME).getTokenOriginalValue();
        name = String.format("%s \"%s\"", typeName, name);
      }
      String message = "{0} has {1} parameters, which is greater than the {2} authorized.";
      getContext().createLineViolation(this, message, node.getFirstChild(PythonGrammar.VARARGSLIST), name, nbParameters, max);
    }
  }
}

