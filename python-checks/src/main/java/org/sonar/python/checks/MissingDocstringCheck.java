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
import com.sonar.sslr.api.Token;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonTokenType;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;

@Rule(
    key = MissingDocstringCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Docstrings should be defined"
)
@SqaleConstantRemediation("5min")
public class MissingDocstringCheck extends PythonCheck {

  public static final String CHECK_KEY = "S1720";

  private static final Pattern EMPTY_STRING_REGEXP = Pattern.compile("([bruBRU]+)?('\\s*')|(\"\\s*\")|('''\\s*''')|(\"\"\"\\s*\"\"\")");
  private static final String MESSAGE_NO_DOCSTRING = "Add a docstring to this %s.";
  private static final String MESSAGE_EMPTY_DOCSTRING = "The docstring for this %s should not be empty.";

  private enum DeclarationType {
    MODULE("module"),
    CLASS("class"),
    METHOD("method"),
    FUNCTION("function");

    private final String value;

    DeclarationType(String value) {
      this.value = value;
    }
  }

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
    checkSimpleStmt(astNode, firstSimpleStmt, DeclarationType.MODULE);
  }

  private void visitClassDef(AstNode astNode) {
    checkFirstSuite(astNode, DeclarationType.CLASS);
  }

  private void visitFuncDef(AstNode astNode) {
    // on methods we check only empty docstrings to avoid false positives on overriding methods
    if (!CheckUtils.isMethodDefinition(astNode)) {
      checkFirstSuite(astNode, DeclarationType.FUNCTION);
    } else {
      checkFirstSuite(astNode, DeclarationType.METHOD);
    }
  }

  private void checkFirstSuite(AstNode astNode, DeclarationType type) {
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
    checkSimpleStmt(astNode, firstSimpleStmt, type);
  }

  private void checkSimpleStmt(AstNode astNode, @Nullable AstNode firstSimpleStmt, DeclarationType type) {
    if (firstSimpleStmt != null) {
      visitFirstStatement(astNode, firstSimpleStmt, type);
    } else {
      raiseIssueNoDocstring(astNode, type);
    }
  }

  private void raiseIssueNoDocstring(AstNode astNode, DeclarationType type) {
    if (type != DeclarationType.METHOD) {
      raiseIssue(astNode, MESSAGE_NO_DOCSTRING, type);
    }
  }

  private void raiseIssue(AstNode astNode, String message, DeclarationType type) {
    String finalMessage = String.format(message, type.value);
    if (type != DeclarationType.MODULE) {
      addIssue(getNameNode(astNode), finalMessage);
    } else {
      getContext().createFileViolation(this, finalMessage);
    }
  }

  private static AstNode getNameNode(AstNode astNode) {
    return astNode.getFirstChild(PythonGrammar.FUNCNAME, PythonGrammar.CLASSNAME);
  }

  private void visitFirstStatement(AstNode astNode, AstNode firstSimpleStmt, DeclarationType type) {
    Token token = firstSimpleStmt.getToken();
    if (token.getType().equals(PythonTokenType.STRING)){
      if (EMPTY_STRING_REGEXP.matcher(token.getValue()).matches()){
        raiseIssue(astNode, MESSAGE_EMPTY_DOCSTRING, type);
      }
    } else {
      raiseIssueNoDocstring(astNode, type);
    }
  }

  private static AstNode firstSimpleStmt(AstNode statement) {
    AstNode stmtList = statement.getFirstChild(PythonGrammar.STMT_LIST);
    if (stmtList != null) {
      return stmtList.getFirstChild(PythonGrammar.SIMPLE_STMT);
    }
    return null;
  }

}
