/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
import com.sonar.sslr.api.Grammar;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.api.PythonGrammar;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;

@Rule(
    key = TooManyParametersCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Functions, methods and lambdas should not have too many parameters",
    tags = Tags.BRAIN_OVERLOAD
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.UNIT_TESTABILITY)
@SqaleConstantRemediation("20min")
@ActivatedByDefault
public class TooManyParametersCheck extends SquidCheck<Grammar> {
  public static final String CHECK_KEY = "S107";

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
      .children(PythonGrammar.TYPEDARGSLIST, PythonGrammar.VARARGSLIST)
      .children(PythonGrammar.TFPDEF, PythonGrammar.FPDEF)
      .size();
    if (nbParameters > max) {
      String name = "Lambda";
      if (node.is(PythonGrammar.FUNCDEF)) {
        String typeName = CheckUtils.isMethodDefinition(node) ? "Method" : "Function";
        name = node.getFirstChild(PythonGrammar.FUNCNAME).getTokenOriginalValue();
        name = String.format("%s \"%s\"", typeName, name);
      }
      String message = "{0} has {1} parameters, which is greater than the {2} authorized.";
      getContext().createLineViolation(this, message, node.getFirstChild(PythonGrammar.TYPEDARGSLIST, PythonGrammar.VARARGSLIST), name, nbParameters, max);
    }
  }
}

