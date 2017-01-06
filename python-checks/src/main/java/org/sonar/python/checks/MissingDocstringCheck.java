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
import java.util.Map;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.DocstringVisitor;
import org.sonar.python.PythonAstScanner;
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
  @SuppressWarnings("unchecked")
  public void visitFile(AstNode file) {
    DocstringVisitor<Grammar> docstringVisitor = new DocstringVisitor<>();
    PythonAstScanner.scanSingleFile(getContext().getFile().getPath(), docstringVisitor);
    
    Map<AstNode, AstNode> docstrings = docstringVisitor.getDocstrings();
    for (Map.Entry<AstNode, AstNode> entry : docstrings.entrySet()) {
      AstNode node = entry.getKey();
      checkSimpleStmt(node, entry.getValue(), getType(node));
    }
  }

  private static DeclarationType getType(AstNode node) {
    if (node.is(PythonGrammar.FILE_INPUT)) {
      return DeclarationType.MODULE;
    } else if (node.is(PythonGrammar.FUNCDEF)) {
      if (CheckUtils.isMethodDefinition(node)) {
        return DeclarationType.METHOD;
      } else {
        return DeclarationType.FUNCTION;
      }
    } else if (node.is(PythonGrammar.CLASSDEF)) {
      return DeclarationType.CLASS;
    }
    return null;
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

}
