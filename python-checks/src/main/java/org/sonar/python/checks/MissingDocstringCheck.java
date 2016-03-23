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
import com.sonar.sslr.api.Token;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonTokenType;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;

import javax.annotation.Nullable;
import java.util.regex.Pattern;

@Rule(
    key = MissingDocstringCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Docstrings should be defined"
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.UNDERSTANDABILITY)
@SqaleConstantRemediation("5min")
public class MissingDocstringCheck extends SquidCheck<Grammar> {

  public static final String CHECK_KEY = "S1720";

  private static final Pattern EMPTY_STRING_REGEXP = Pattern.compile("([bruBRU]+)?('\\s*')|(\"\\s*\")|('''\\s*''')|(\"\"\"\\s*\"\"\")");
  private static final String MESSAGE_NO_DOCSTRING = "Add a docstring to this %s.";
  private static final String MESSAGE_EMPTY_DOCSTRING = "The docstring for this %s should not be empty.";

  @Override
  public void init() {
    subscribeTo(PythonGrammar.FILE_INPUT, PythonGrammar.FUNCDEF, PythonGrammar.CLASSDEF);
  }

  @Override
  public void visitNode(AstNode astNode) {
    if (astNode.is(PythonGrammar.FILE_INPUT)) {
      visitModule(astNode);
    }
    if (astNode.is(PythonGrammar.FUNCDEF)) {
      visitFuncDef(astNode);
    }
    if (astNode.is(PythonGrammar.CLASSDEF)) {
      visitClassDef(astNode);
    }
  }

  private void visitModule(AstNode astNode) {
    AstNode firstStatement = astNode.getFirstChild(PythonGrammar.STATEMENT);
    AstNode firstSimpleStmt = null;
    if (firstStatement != null) {
      firstSimpleStmt = firstSimpleStmt(firstStatement);
    }
    checkSimpleStmt(astNode, firstSimpleStmt, "module");
  }

  private void visitClassDef(AstNode astNode) {
    checkFirstSuite(astNode, "class");
  }

  private void visitFuncDef(AstNode astNode) {
    // on methods we check only empty docstrings to avoid false positives on overriding methods
    if (!CheckUtils.isMethodDefinition(astNode)) {
      checkFirstSuite(astNode, "function");
    } else {
      checkFirstSuite(astNode, "method");
    }
  }

  private void checkFirstSuite(AstNode astNode, String typeName) {
    AstNode suite = astNode.getFirstChild(PythonGrammar.SUITE);
    AstNode firstStatement = suite.getFirstChild(PythonGrammar.STATEMENT);
    AstNode firstSimpleStmt;
    if (firstStatement == null) {
      firstSimpleStmt = suite
        .getFirstChild(PythonGrammar.STMT_LIST)
        .getFirstChild(PythonGrammar.SIMPLE_STMT);
    } else {
      firstSimpleStmt = firstSimpleStmt(firstStatement);
    }
    checkSimpleStmt(astNode, firstSimpleStmt, typeName);
  }

  private void checkSimpleStmt(AstNode astNode, @Nullable AstNode firstSimpleStmt, String typeName) {
    if (firstSimpleStmt != null) {
      visitFirstStatement(astNode, firstSimpleStmt, typeName);
    } else {
      raiseIssueNoDocstring(astNode, typeName);
    }
  }

  private void raiseIssueNoDocstring(AstNode astNode, String typeName) {
    if (!"method".equals(typeName)) {
      getContext().createLineViolation(this, String.format(MESSAGE_NO_DOCSTRING, typeName), astNode);
    }
  }

  private void visitFirstStatement(AstNode astNode, AstNode firstSimpleStmt, String typeName) {
    Token token = firstSimpleStmt.getToken();
    if (token.getType().equals(PythonTokenType.STRING)){
      if (EMPTY_STRING_REGEXP.matcher(token.getValue()).matches()){
        getContext().createLineViolation(this, String.format(MESSAGE_EMPTY_DOCSTRING, typeName), astNode);
      }
    } else {
      raiseIssueNoDocstring(astNode, typeName);
    }
  }

  private AstNode firstSimpleStmt(AstNode statement) {
    AstNode stmtList = statement.getFirstChild(PythonGrammar.STMT_LIST);
    if (stmtList != null) {
      return stmtList.getFirstChild(PythonGrammar.SIMPLE_STMT);
    }
    return null;
  }

}
