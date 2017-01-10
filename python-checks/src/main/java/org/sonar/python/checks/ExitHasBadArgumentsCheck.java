/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2017 SonarSource SA
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
import java.util.List;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;

@Rule(
    key = ExitHasBadArgumentsCheck.CHECK_KEY,
    priority = Priority.CRITICAL,
    name = "\"__exit__\" should accept type, value, and traceback arguments",
    tags = Tags.BUG
)
@SqaleConstantRemediation("5min")
@ActivatedByDefault
public class ExitHasBadArgumentsCheck extends PythonCheck {

  public static final String MESSAGE_ADD = "Add the missing argument.";
  public static final String MESSAGE_REMOVE = "Remove the unnecessary argument.";

  private static final int EXIT_ARGUMENTS_NUMBER = 4;

  public static final String CHECK_KEY = "S2733";

  @Override
  public void init() {
    subscribeTo(PythonGrammar.FUNCDEF);
  }

  @Override
  public void visitNode(AstNode node) {
    if (!"__exit__".equals(node.getFirstChild(PythonGrammar.FUNCNAME).getToken().getValue())){
      return;
    }
    AstNode varArgList = node.getFirstChild(PythonGrammar.TYPEDARGSLIST);
    if (varArgList != null) {
      List<AstNode> arguments = varArgList.getChildren(PythonGrammar.TFPDEF);
      for (AstNode argument : arguments) {
        if (argument.getPreviousSibling() != null && argument.getPreviousSibling().is(PythonPunctuator.MUL_MUL, PythonPunctuator.MUL)) {
          return;
        }
      }
      raiseIssue(node, arguments.size());
    } else {
      raiseIssue(node, 0);
    }
  }

  private void raiseIssue(AstNode node, int argumentsNumber) {
    if (argumentsNumber != EXIT_ARGUMENTS_NUMBER){
      String message = MESSAGE_ADD;
      if (argumentsNumber > EXIT_ARGUMENTS_NUMBER){
        message = MESSAGE_REMOVE;
      }
      AstNode funcName = node.getFirstChild(PythonGrammar.FUNCNAME);
      AstNode rightParenthesis = node.getFirstChild(PythonPunctuator.RPARENTHESIS);
      addIssue(new IssueLocation(funcName, rightParenthesis, message));
    }
  }
}

