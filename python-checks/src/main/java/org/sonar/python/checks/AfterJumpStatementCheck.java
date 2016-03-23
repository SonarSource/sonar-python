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
    key = AfterJumpStatementCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Jump statements should not be followed by other statements",
    tags = {Tags.UNUSED, Tags.CERT, Tags.CWE, Tags.MISRA}
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.LOGIC_RELIABILITY)
@SqaleConstantRemediation("5min")
@ActivatedByDefault
public class AfterJumpStatementCheck extends SquidCheck<Grammar> {

  public static final String CHECK_KEY = "S1763";

  private static final String MESSAGE = "Remove the code after this \"%s\".";

  @Override
  public void init() {
    subscribeTo(
        PythonGrammar.RETURN_STMT,
        PythonGrammar.RAISE_STMT,
        PythonGrammar.BREAK_STMT,
        PythonGrammar.CONTINUE_STMT
    );
  }

  @Override
  public void visitNode(AstNode node) {
    AstNode simpleStatement = node.getParent();

    AstNode nextSibling = simpleStatement.getNextSibling();
    if (nextSibling != null && nextSibling.getNextSibling() != null) {
      raiseIssue(node);
      return;
    }

    AstNode stmtList = simpleStatement.getParent();
    if (stmtList.getParent().is(PythonGrammar.STATEMENT)){
      nextSibling = stmtList.getParent().getNextSibling();
      if (nextSibling != null && nextSibling.getNextSibling() != null){
        raiseIssue(node);
      }
    }
  }

  private void raiseIssue(AstNode node) {
    getContext().createLineViolation(this, String.format(MESSAGE, node.getTokenValue()), node);
  }

}

