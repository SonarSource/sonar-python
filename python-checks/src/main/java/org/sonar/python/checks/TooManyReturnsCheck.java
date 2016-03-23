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
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;
import org.sonar.sslr.ast.AstSelect;

@Rule(
    key = TooManyReturnsCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Functions should not contain too many return statements",
    tags = Tags.BRAIN_OVERLOAD
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.UNDERSTANDABILITY)
@SqaleConstantRemediation("20min")
public class TooManyReturnsCheck extends SquidCheck<Grammar> {
  public static final String CHECK_KEY = "S1142";

  private static final int DEFAULT_MAX = 3;
  private static final String MESSAGE = "This function has %s returns or yields, which is more than the %s allowed.";

  @RuleProperty(key = "max", defaultValue = "" + DEFAULT_MAX)
  public int max = DEFAULT_MAX;

  @Override
  public void init() {
    subscribeTo(PythonGrammar.FUNCDEF);
  }

  @Override
  public void visitNode(AstNode node) {
    AstSelect returnStatements = node.select().descendants(PythonGrammar.RETURN_STMT, PythonGrammar.YIELD_STMT);
    int returnCount = 0;
    for (AstNode returnStatement : returnStatements){
      if (CheckUtils.insideFunction(returnStatement, node)){
        returnCount++;
      }
    }
    if (returnCount > max) {
      getContext().createLineViolation(this, String.format(MESSAGE, returnCount, max), node.getFirstChild(PythonGrammar.FUNCNAME));
    }
  }
}

