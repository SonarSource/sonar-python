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
import com.sonar.sslr.api.Token;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonTokenType;
import org.sonar.squidbridge.checks.SquidCheck;

import java.util.regex.Pattern;

@Rule(
  key = "S1720",
  priority = Priority.MAJOR)
public class MissingDocstringCheck extends SquidCheck<Grammar> {

  private static final Pattern EMPTY_STRING_REGEXP =
    Pattern.compile("([brBR]+)?('\\s*')|(\"\\s*\")|('''\\s*''')|(\"\"\"\\s*\"\"\")");

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
    if (!CheckUtils.isMethodDefinition(astNode)) {
      checkFirstSuite(astNode, "function");
    }
  }

  private void checkFirstSuite(AstNode astNode, String typeName) {
    AstNode suite = astNode.getFirstChild(PythonGrammar.SUITE);
    AstNode firstStatement = suite.getFirstChild(PythonGrammar.STATEMENT);
    AstNode firstSimpleStmt = null;
    if (firstStatement == null) {
      firstSimpleStmt = suite
        .getFirstChild(PythonGrammar.STMT_LIST)
        .getFirstChild(PythonGrammar.SIMPLE_STMT);
    } else {
      firstSimpleStmt = firstSimpleStmt(firstStatement);
    }
    checkSimpleStmt(astNode, firstSimpleStmt, typeName);
  }

  private void checkSimpleStmt(AstNode astNode, AstNode firstSimpleStmt, String typeName) {
    if (firstSimpleStmt != null) {
      Token token = firstSimpleStmt.getToken();
      if (isNonEmptyString(token)) {
        return;
      }
    }
    getContext().createLineViolation(this, "Add a docstring to this " + typeName, astNode);
  }

  private AstNode firstSimpleStmt(AstNode statement) {
    AstNode stmtList = statement.getFirstChild(PythonGrammar.STMT_LIST);
    if (stmtList != null) {
      return stmtList.getFirstChild(PythonGrammar.SIMPLE_STMT);
    }
    return null;
  }

  private boolean isNonEmptyString(Token token) {
    return token.getType() == PythonTokenType.STRING
      && !EMPTY_STRING_REGEXP.matcher(token.getValue()).matches();
  }
}
