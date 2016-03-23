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
    key = BreakContinueOutsideLoopCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "\"break\" and \"continue\" should not be used outside a loop"
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.LANGUAGE_RELATED_PORTABILITY)
@SqaleConstantRemediation("10min")
@ActivatedByDefault
public class BreakContinueOutsideLoopCheck extends SquidCheck<Grammar> {

  public static final String MESSAGE = "Remove this \"%s\" statement";
  public static final String CHECK_KEY = "S1716";

  @Override
  public void init() {
    subscribeTo(PythonGrammar.BREAK_STMT, PythonGrammar.CONTINUE_STMT);
  }

  @Override
  public void visitNode(AstNode node) {
    AstNode currentParent = node.getParent();
    while (currentParent != null){
      if (currentParent.is(PythonGrammar.WHILE_STMT, PythonGrammar.FOR_STMT)){
        return;
      } else if (currentParent.is(PythonGrammar.FUNCDEF, PythonGrammar.CLASSDEF)){
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

