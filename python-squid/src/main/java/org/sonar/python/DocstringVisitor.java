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
package org.sonar.python;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstVisitor;
import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.api.Token;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonTokenType;
import org.sonar.squidbridge.SquidAstVisitor;

/**
 * Visitor that retrieves all docstrings from a Python file.
 * <p>
 * Reminder: a docstring is a string literal that occurs as the first statement
 * in a module, function, class, or method definition.
 */
public class DocstringVisitor<G extends Grammar> extends SquidAstVisitor<G> implements AstVisitor {

  /**
   * Key = an AstNode (module, function, class or method).
   * Value = the docstring of the node. Can be null
   */
  private Map<AstNode, AstNode> docstrings = new HashMap<>();

  public Map<AstNode, AstNode> getDocstrings() {
    return docstrings;
  }

  @Override
  public void init() {
    subscribeTo(PythonGrammar.FILE_INPUT, PythonGrammar.FUNCDEF, PythonGrammar.CLASSDEF);
    docstrings.clear();
  }

  @Override
  public void visitNode(AstNode astNode) {
    if (astNode.is(PythonGrammar.FILE_INPUT)) {
      visitModule(astNode);
    } else if (astNode.is(PythonGrammar.FUNCDEF)) {
      visitFuncDef(astNode);
    } else if (astNode.is(PythonGrammar.CLASSDEF)) {
      visitClassDef(astNode);
    }
  }

  private void visitModule(AstNode astNode) {
    docstrings.put(astNode, null);

    AstNode firstStatement = astNode.getFirstChild(PythonGrammar.STATEMENT);
    AstNode firstSimpleStmt = null;
    if (firstStatement != null) {
      firstSimpleStmt = getFirstSimpleStmt(firstStatement);
    }
    visitSimpleStmt(astNode, firstSimpleStmt);
  }

  private void visitFuncDef(AstNode astNode) {
    docstrings.put(astNode, null);
    visitFirstSuite(astNode);
  }

  private void visitClassDef(AstNode astNode) {
    docstrings.put(astNode, null);
    visitFirstSuite(astNode);
  }

  private void visitFirstSuite(AstNode astNode) {
    AstNode suite = astNode.getFirstChild(PythonGrammar.SUITE);
    AstNode firstStatement = suite.getFirstChild(PythonGrammar.STATEMENT);
    AstNode firstSimpleStmt;
    if (firstStatement == null) {
      firstSimpleStmt = suite.getFirstChild(PythonGrammar.STMT_LIST).getFirstChild(PythonGrammar.SIMPLE_STMT);
    } else {
      firstSimpleStmt = getFirstSimpleStmt(firstStatement);
    }
    visitSimpleStmt(astNode, firstSimpleStmt);
  }

  private void visitSimpleStmt(AstNode astNode, @Nullable AstNode firstSimpleStmt) {
    if (firstSimpleStmt != null) {
      visitFirstStatement(astNode, firstSimpleStmt);
    }
  }

  private void visitFirstStatement(AstNode astNode, AstNode firstSimpleStmt) {
    Token token = firstSimpleStmt.getToken();
    if (token.getType().equals(PythonTokenType.STRING)) {
      docstrings.put(astNode, firstSimpleStmt);
    }
  }

  private static AstNode getFirstSimpleStmt(AstNode statement) {
    AstNode stmtList = statement.getFirstChild(PythonGrammar.STMT_LIST);
    if (stmtList != null) {
      return stmtList.getFirstChild(PythonGrammar.SIMPLE_STMT);
    }
    return null;
  }

}
