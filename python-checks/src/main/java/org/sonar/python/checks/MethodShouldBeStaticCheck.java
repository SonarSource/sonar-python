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
import java.util.List;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonTokenType;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.checks.SquidCheck;

@Rule(
    key = MethodShouldBeStaticCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Methods that don't access instance data should be \"static\"",
    tags = {Tags.PERFORMANCE}
)
@SqaleConstantRemediation("5min")
@ActivatedByDefault
public class MethodShouldBeStaticCheck extends SquidCheck<Grammar> {

  public static final String CHECK_KEY = "S2325";

  private static final String MESSAGE = "Make this method static.";

  @Override
  public void init() {
    subscribeTo(PythonGrammar.FUNCDEF);
  }

  @Override
  public void visitNode(AstNode node) {
    if (CheckUtils.isMethodDefinition(node) && !alreadyStaticMethod(node) && hasValuableCode(node)){
      String self = getFirstArgument(node);
      if (self != null && !isUsed(node, self) && !onlyRaisesNotImplementedError(node)){
        getContext().createLineViolation(this, MESSAGE, node.getFirstChild(PythonGrammar.FUNCNAME));
      }
    }
  }

  private static boolean onlyRaisesNotImplementedError(AstNode funcDef) {
    AstNode suite = funcDef.getFirstChild(PythonGrammar.SUITE);
    List<AstNode> statements = suite.getChildren(PythonGrammar.STATEMENT);

    if (statements.isEmpty()) {
      // case of a method defined in one single line
      AstNode statementList = suite.getFirstChild();
      if (raisesNotImplementedError(statementList)) {
        return true;
      }
      
    } else {
      // standard case of a method defined on more than one line
      if (statements.size() <= 2) {
        if (statements.size() == 2 && !isDocstring(statements.get(0))) {
          return false;
        }
  
        AstNode statementList = statements.get(statements.size() - 1).getFirstChild(PythonGrammar.STMT_LIST);
        if (statementList != null && raisesNotImplementedError(statementList)) {
          return true;
        }
      }
    }

    return false;
  }
  
  private static boolean raisesNotImplementedError(AstNode statementList) {
    if (isRaise(statementList)) {
      AstNode testStatement = statementList.getFirstDescendant(PythonGrammar.TEST);
      if ("NotImplementedError".equals(testStatement.getToken().getValue())) {
        return true;
      }
    }
    return false;
  }

  private static boolean hasValuableCode(AstNode funcDef) {
    AstNode statementList = funcDef.getFirstChild(PythonGrammar.SUITE).getFirstChild(PythonGrammar.STMT_LIST);
    if (statementList != null && statementList.getChildren(PythonGrammar.SIMPLE_STMT).size() == 1) {
      return !statementList.getFirstChild(PythonGrammar.SIMPLE_STMT).getFirstChild().is(PythonGrammar.PASS_STMT);
    }

    List<AstNode> statements = funcDef.getFirstChild(PythonGrammar.SUITE).getChildren(PythonGrammar.STATEMENT);
    if (statements.size() == 1) {
      return !isDocstringOrPass(statements.get(0));
    }

    return statements.size() != 2 || !isDocstringAndPass(statements.get(0), statements.get(1));
  }

  private static boolean isDocstring(AstNode statement) {
    return statement.getToken().getType().equals(PythonTokenType.STRING);
  }

  private static boolean isDocstringOrPass(AstNode statement) {
    return statement.getFirstDescendant(PythonGrammar.PASS_STMT) != null || statement.getToken().getType().equals(PythonTokenType.STRING);
  }
  
  private static boolean isDocstringAndPass(AstNode statement1, AstNode statement2) {
    return statement1.getToken().getType().equals(PythonTokenType.STRING) && statement2.getFirstDescendant(PythonGrammar.PASS_STMT) != null;
  }
  
  private static boolean isRaise(AstNode statementList) {
    return statementList.getFirstChild().hasDescendant(PythonGrammar.RAISE_STMT);
  }

  private static boolean isUsed(AstNode funcDef, String self) {
    List<AstNode> names = funcDef.getFirstChild(PythonGrammar.SUITE).getDescendants(PythonGrammar.NAME);
    for (AstNode name : names){
      if (name.getTokenValue().equals(self)){
        return true;
      }
    }
    return false;
  }

  private static String getFirstArgument(AstNode funcDef) {
    AstNode argList = funcDef.getFirstChild(PythonGrammar.TYPEDARGSLIST);
    if (argList != null){
      return argList.getFirstDescendant(PythonGrammar.NAME).getTokenValue();
    } else {
      return null;
    }
  }

  private static boolean alreadyStaticMethod(AstNode funcDef) {
    AstNode decorators = funcDef.getFirstChild(PythonGrammar.DECORATORS);
    if (decorators != null){
      List<AstNode> decoratorList = decorators.getChildren(PythonGrammar.DECORATOR);
      for (AstNode decorator : decoratorList){
        AstNode name = decorator.getFirstDescendant(PythonGrammar.NAME);
        if (name != null && ("staticmethod".equals(name.getTokenValue()) || "classmethod".equals(name.getTokenValue()))){
          return true;
        }
      }
    }
    return false;
  }

}

