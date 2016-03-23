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
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.Token;
import org.sonar.python.api.PythonGrammar;

import java.util.LinkedList;
import java.util.List;

public class NewSymbolsAnalyzer {
  private List<Token> symbols;

  public List<Token> getClassFields(AstNode classDef) {
    symbols = new LinkedList<>();
    findFieldsInClassBody(classDef);

    List<AstNode> methods = classDef.getFirstChild(PythonGrammar.SUITE).getDescendants(PythonGrammar.FUNCDEF);

    for (AstNode method : methods) {
      addFieldsInMethod(method);
    }
    return symbols;
  }

  private void addFieldsInMethod(AstNode method) {
    AstNode suite = method.getFirstChild(PythonGrammar.SUITE);
    List<AstNode> expressions = suite.getDescendants(PythonGrammar.EXPRESSION_STMT);
    for (AstNode expression : expressions) {
      if (CheckUtils.isAssignmentExpression(expression)) {
        addSelfDotIdentifiersFromLongAssignmentExpression(expression);
      }
    }
  }

  private List<AstNode> getTestsFromLongAssignmentExpression(AstNode expression) {
    List<AstNode> assignedExpressions = expression.getChildren(PythonGrammar.TESTLIST_STAR_EXPR);
    assignedExpressions.remove(assignedExpressions.size() - 1);
    List<AstNode> tests = new LinkedList<>();
    for (AstNode assignedExpression : assignedExpressions) {
      tests.addAll(assignedExpression.getDescendants(PythonGrammar.TEST));
    }
    return tests;
  }

  public void addSelfDotIdentifiersFromLongAssignmentExpression(AstNode expression) {
    List<AstNode> tests = getTestsFromLongAssignmentExpression(expression);
    for (AstNode test : tests) {
      addSelfField(test);
    }
  }

  public void addSimpleIdentifiersFromLongAssignmentExpression(AstNode expression) {
    List<AstNode> tests = getTestsFromLongAssignmentExpression(expression);
    for (AstNode test : tests) {
      addSimpleField(test);
    }
  }

  private void addSelfField(AstNode test) {
    if ("self".equals(test.getTokenValue())) {
      AstNode trailer = test.getFirstDescendant(PythonGrammar.TRAILER);
      if (trailer != null && trailer.getFirstChild(PythonGrammar.NAME) != null) {
        Token token = trailer.getFirstChild(PythonGrammar.NAME).getToken();
        if (!CheckUtils.containsValue(symbols, token.getValue())) {
          symbols.add(token);
        }
      }
    }
  }

  private void addSimpleField(AstNode test) {
    Token token = test.getToken();
    if (test.getNumberOfChildren() == 1
        && test.getFirstChild().is(PythonGrammar.ATOM)
        && token.getType().equals(GenericTokenType.IDENTIFIER) && !CheckUtils.containsValue(symbols, token.getValue())) {
      symbols.add(token);
    }
  }

  private List<Token> findFieldsInClassBody(AstNode classDef) {
    List<AstNode> statements = classDef.getFirstChild(PythonGrammar.SUITE).getChildren(PythonGrammar.STATEMENT);
    List<AstNode> expressions = new LinkedList<>();
    for (AstNode statement : statements) {
      if (!statement.hasDescendant(PythonGrammar.FUNCDEF)) {
        expressions.addAll(statement.getDescendants(PythonGrammar.EXPRESSION_STMT));
      }
    }
    for (AstNode expression : expressions) {
      if (CheckUtils.isAssignmentExpression(expression)) {
        addSimpleIdentifiersFromLongAssignmentExpression(expression);
      }
    }
    return symbols;
  }

  public List<Token> getVariablesFromLongAssignmentExpression(List<Token> varNames, AstNode expression) {
    symbols = varNames;
    addSimpleIdentifiersFromLongAssignmentExpression(expression);
    return symbols;
  }
}
