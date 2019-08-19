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
package org.sonar.python.api.tree;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Token;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.tree.PyElseStatementTreeImpl;
import org.sonar.python.tree.PyExpressionTreeImpl;
import org.sonar.python.tree.PyFileInputTreeImpl;
import org.sonar.python.tree.PyIfStatementTreeImpl;

public class PythonTreeMaker {

  public PyFileInputTree fileInput(AstNode astNode) {
    List<PyStatementTree> statements = astNode.getChildren(PythonGrammar.STATEMENT).stream().map(this::statement).collect(Collectors.toList());
    return new PyFileInputTreeImpl(astNode, statements);
  }

  private PyStatementTree statement(AstNode astNode) {
    if (astNode.is(PythonGrammar.IF_STMT)) {
      return ifStatement(astNode);
    }
    return null;
  }

  public PyIfStatementTree ifStatement(AstNode astNode) {
    Token ifToken = astNode.getTokens().get(0);
    AstNode condition = astNode.getFirstChild(PythonGrammar.TEST);
    AstNode suite = astNode.getFirstChild(PythonGrammar.SUITE);
    List<PyStatementTree> statements = suite.getChildren(PythonGrammar.STATEMENT).stream().map(this::statement).collect(Collectors.toList());
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
    List<PyStatementTree> statements = suite.getChildren(PythonGrammar.STATEMENT).stream().map(this::statement).collect(Collectors.toList());
    return new PyIfStatementTreeImpl(
      astNode, elifToken, expression(condition), statements, Collections.emptyList(), null);
  }

  private PyElseStatementTree elseStatement(AstNode astNode) {
    Token elseToken = astNode.getPreviousSibling().getPreviousSibling().getToken();
    List<PyStatementTree> statements = astNode.getChildren(PythonGrammar.STATEMENT).stream().map(this::statement).collect(Collectors.toList());
    return new PyElseStatementTreeImpl(astNode, elseToken, statements);
  }

  PyExpressionTree expression(AstNode astNode) {
    return new PyExpressionTreeImpl(astNode);
  }


}
