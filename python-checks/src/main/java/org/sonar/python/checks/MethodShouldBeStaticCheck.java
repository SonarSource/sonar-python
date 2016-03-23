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
import org.sonar.python.api.PythonTokenType;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;

import java.util.List;

@Rule(
    key = MethodShouldBeStaticCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Methods that don't access instance data should be \"static\"",
    tags = {Tags.PERFORMANCE}
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.CPU_EFFICIENCY)
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
      if (self != null && !isUsed(node, self)){
        getContext().createLineViolation(this, MESSAGE, node.getFirstChild(PythonGrammar.FUNCNAME));
      }
    }
  }

  private boolean hasValuableCode(AstNode funcDef) {
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

  private boolean isDocstringOrPass(AstNode statement) {
    return statement.getFirstDescendant(PythonGrammar.PASS_STMT) != null || statement.getToken().getType().equals(PythonTokenType.STRING);
  }

  private boolean isDocstringAndPass(AstNode statement1, AstNode statement2) {
    return statement1.getToken().getType().equals(PythonTokenType.STRING) && statement2.getFirstDescendant(PythonGrammar.PASS_STMT) != null;
  }

  private boolean isUsed(AstNode funcDef, String self) {
    List<AstNode> names = funcDef.getFirstChild(PythonGrammar.SUITE).getDescendants(PythonGrammar.NAME);
    for (AstNode name : names){
      if (name.getTokenValue().equals(self)){
        return true;
      }
    }
    return false;
  }

  private String getFirstArgument(AstNode funcDef) {
    AstNode argList = funcDef.getFirstChild(PythonGrammar.TYPEDARGSLIST);
    if (argList != null){
      return argList.getFirstDescendant(PythonGrammar.NAME).getTokenValue();
    } else {
      return null;
    }
  }

  private boolean alreadyStaticMethod(AstNode funcDef) {
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

