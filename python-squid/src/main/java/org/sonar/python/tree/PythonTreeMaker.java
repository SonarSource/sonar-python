/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.tree;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Token;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.tree.PyElseStatementTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyFileInputTree;
import org.sonar.python.api.tree.PyIfStatementTree;
import org.sonar.python.api.tree.PyStatementTree;

public class PythonTreeMaker {

  public PyFileInputTree fileInput(AstNode astNode) {
    List<PyStatementTree> statements = getStatements(astNode).stream().map(this::statement).collect(Collectors.toList());
    return new PyFileInputTreeImpl(astNode, statements);
  }

  private PyStatementTree statement(AstNode astNode) {
    if (astNode.is(PythonGrammar.IF_STMT)) {
      return ifStatement(astNode);
    }
    return null;
  }

  private List<AstNode> getStatementsFromSuite(AstNode astNode) {
    if (astNode.is(PythonGrammar.SUITE)) {
      List<AstNode> statements = getStatements(astNode);
      if (statements.isEmpty()) {
        AstNode stmtListNode = astNode.getFirstChild(PythonGrammar.STMT_LIST);
        return stmtListNode.getChildren(PythonGrammar.SIMPLE_STMT).stream()
          .map(AstNode::getFirstChild)
          .collect(Collectors.toList());
      }
      return statements;
    }
    return Collections.emptyList();
  }

  private List<AstNode> getStatements(AstNode astNode) {
    List<AstNode> statements = astNode.getChildren(PythonGrammar.STATEMENT);
    return statements.stream().flatMap(stmt -> {
      if (stmt.hasDirectChildren(PythonGrammar.STMT_LIST)) {
        AstNode stmtListNode = stmt.getFirstChild(PythonGrammar.STMT_LIST);
        return stmtListNode.getChildren(PythonGrammar.SIMPLE_STMT).stream()
          .map(AstNode::getFirstChild);
      }
      return stmt.getChildren(PythonGrammar.COMPOUND_STMT).stream()
        .map(AstNode::getFirstChild);
    }).collect(Collectors.toList());
  }

  public PyIfStatementTree ifStatement(AstNode astNode) {
    Token ifToken = astNode.getTokens().get(0);
    AstNode condition = astNode.getFirstChild(PythonGrammar.TEST);
    AstNode suite = astNode.getFirstChild(PythonGrammar.SUITE);
    List<PyStatementTree> statements = getStatementsFromSuite(suite).stream().map(this::statement).collect(Collectors.toList());
    AstNode elseSuite = astNode.getLastChild(PythonGrammar.SUITE);
    PyElseStatementTree elseStatement = null;
    if (elseSuite.getPreviousSibling().getPreviousSibling().is(PythonKeyword.ELSE)) {
      elseStatement = elseStatement(elseSuite);
    }
    List<PyIfStatementTree> elifBranches = astNode.getChildren(PythonKeyword.ELIF).stream()
      .map(this::elifStatement)
      .collect(Collectors.toList());

    return new PyIfStatementTreeImpl(
      astNode, ifToken, expression(condition), statements, elifBranches, elseStatement);
  }

  private PyIfStatementTree elifStatement(AstNode astNode) {
    Token elifToken = astNode.getToken();
    AstNode suite = astNode.getNextSibling().getNextSibling().getNextSibling();
    AstNode condition = astNode.getNextSibling();
    List<PyStatementTree> statements = getStatementsFromSuite(suite).stream().map(this::statement).collect(Collectors.toList());
    return new PyIfStatementTreeImpl(
      astNode, elifToken, expression(condition), statements);
  }

  private PyElseStatementTree elseStatement(AstNode astNode) {
    Token elseToken = astNode.getPreviousSibling().getPreviousSibling().getToken();
    List<PyStatementTree> statements = getStatementsFromSuite(astNode).stream().map(this::statement).collect(Collectors.toList());
    return new PyElseStatementTreeImpl(astNode, elseToken, statements);
  }

  PyExpressionTree expression(AstNode astNode) {
    return new PyExpressionTreeImpl(astNode);
  }


}
