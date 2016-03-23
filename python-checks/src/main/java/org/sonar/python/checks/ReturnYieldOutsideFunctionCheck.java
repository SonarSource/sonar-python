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
import org.sonar.python.api.PythonGrammar;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;

@Rule(
    key = ReturnYieldOutsideFunctionCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "\"yield\" and \"return\" should not be used outside functions",
    tags = Tags.BUG
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.INSTRUCTION_RELIABILITY)
@SqaleConstantRemediation("15min")
@ActivatedByDefault
public class ReturnYieldOutsideFunctionCheck extends SquidCheck<Grammar> {

  public static final String MESSAGE = "Remove this use of \"%s\".";
  public static final String CHECK_KEY = "S2711";

  @Override
  public void init() {
    subscribeTo(PythonGrammar.RETURN_STMT, PythonGrammar.YIELD_STMT);
  }

  @Override
  public void visitNode(AstNode node) {
    AstNode currentParent = node.getParent();
    while (currentParent != null){
      if (currentParent.is(PythonGrammar.FUNCDEF)){
        return;
      } else if (currentParent.is(PythonGrammar.CLASSDEF)){
        raiseIssue(node);
        return;
      }
      currentParent = currentParent.getParent();
    }
    raiseIssue(node);
  }

  private void raiseIssue(AstNode node) {
    getContext().createLineViolation(this, String.format(MESSAGE, node.getToken().getValue()), node);
  }
}
