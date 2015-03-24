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
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.Token;
import org.sonar.python.api.PythonGrammar;

import java.util.LinkedList;
import java.util.List;

public class NewSymbolsAnalyzer {
  private static List<Token> symbols;

  public static List<Token> getClassFields(AstNode classDef){
    symbols = new LinkedList<>();
    findFieldsInClassBody(classDef);

    List<AstNode> methods = classDef.getFirstChild(PythonGrammar.SUITE).getDescendants(PythonGrammar.FUNCDEF);

    for (AstNode method : methods){
      addFieldsInMethod(method);
    }
    return symbols;
  }

  private static void addFieldsInMethod(AstNode method) {
    AstNode suite = method.getFirstChild(PythonGrammar.SUITE);
    List<AstNode> expressions = suite.getDescendants(PythonGrammar.EXPRESSION_STMT);
    for (AstNode expression : expressions) {
      if (CheckUtils.isAssignmentExpression(expression)) {
        addIdentifiersFromLongAssignmentExpression(expression, true);
      }
    }
  }

  public static void addIdentifiersFromLongAssignmentExpression(AstNode expression, boolean withSelf) {
    List<AstNode> assignedExpressions = expression.getChildren(PythonGrammar.TESTLIST_STAR_EXPR);
    assignedExpressions.remove(assignedExpressions.size() - 1);
    List<AstNode> tests = new LinkedList<>();
    for (AstNode assignedExpression : assignedExpressions){
      tests.addAll(assignedExpression.getDescendants(PythonGrammar.TEST));
    }
    for (AstNode test : tests) {
      if (withSelf){
        addSelfField(test);
      } else {
        addSimpleField(test);
      }
    }
  }

  private static void addSelfField(AstNode test) {
    if ("self".equals(test.getTokenValue())){
      AstNode trailer = test.getFirstDescendant(PythonGrammar.TRAILER);
      if (trailer != null && trailer.getFirstChild(PythonGrammar.NAME) != null){
        Token token = trailer.getFirstChild(PythonGrammar.NAME).getToken();
        if (!CheckUtils.containsValue(symbols, token.getValue())) {
          symbols.add(token);
        }
      }
    }
  }

  private static void addSimpleField(AstNode test) {
    Token token = test.getToken();
    if (test.getNumberOfChildren() == 1
        && test.getFirstChild().is(PythonGrammar.ATOM)
        && token.getType().equals(GenericTokenType.IDENTIFIER) && !CheckUtils.containsValue(symbols, token.getValue())){
      symbols.add(token);
    }
  }

  private static List<Token> findFieldsInClassBody(AstNode classDef) {
    List<AstNode> statements = classDef.getFirstChild(PythonGrammar.SUITE).getChildren(PythonGrammar.STATEMENT);
    List<AstNode> expressions = new LinkedList<>();
    for (AstNode statement : statements){
      if (!statement.hasDescendant(PythonGrammar.FUNCDEF)){
        expressions.addAll(statement.getDescendants(PythonGrammar.EXPRESSION_STMT));
      }
    }
    for (AstNode expression : expressions) {
      if (CheckUtils.isAssignmentExpression(expression)){
        addIdentifiersFromLongAssignmentExpression(expression, false);
      }
    }
    return symbols;
  }

  public static List<Token> getVariablesFromLongAssignmentExpression(List<Token> varNames, AstNode expression) {
    symbols = varNames;
    addIdentifiersFromLongAssignmentExpression(expression, false);
    return symbols;
  }
}
