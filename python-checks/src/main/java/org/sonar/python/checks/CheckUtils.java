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
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;

import java.util.LinkedList;
import java.util.List;

public class CheckUtils {

  private CheckUtils() {

  }

  public static boolean isMethodDefinition(AstNode node) {
    AstNode parent = node.getParent();
    for (int i = 0; i < 3; i++) {
      if (parent != null) {
        parent = parent.getParent();
      }
    }
    return parent != null && parent.is(PythonGrammar.CLASSDEF);
  }

  public static boolean equalNodes(AstNode node1, AstNode node2){
    if (!node1.getType().equals(node2.getType()) || node1.getNumberOfChildren() != node2.getNumberOfChildren()){
      return false;
    }

    if (node1.getNumberOfChildren() == 0) {
      return node1.is(PythonTokenType.INDENT, PythonTokenType.DEDENT) || node1.getToken().getValue().equals(node2.getToken().getValue());
    }

    List<AstNode> children1 = node1.getChildren();
    List<AstNode> children2 = node2.getChildren();
    for (int i = 0; i < children1.size(); i++){
      if (!equalNodes(children1.get(i), children2.get(i))){
        return false;
      }
    }
    return true;
  }

  public static boolean insideFunction(AstNode astNode, AstNode funcDef) {
    return astNode.getFirstAncestor(PythonGrammar.FUNCDEF).equals(funcDef);
  }

  public static List<Token> getClassFields(AstNode classDef){
    List<Token> allFields = findFieldsInClassBody(classDef);

    List<AstNode> methods = classDef.getFirstChild(PythonGrammar.SUITE).getDescendants(PythonGrammar.FUNCDEF);

    for (AstNode method : methods){
      addFieldsInMethod(allFields, method);
    }
    return allFields;
  }

  private static void addFieldsInMethod(List<Token> allFields, AstNode method) {
    AstNode suite = method.getFirstChild(PythonGrammar.SUITE);
    List<AstNode> expressions = suite.getDescendants(PythonGrammar.EXPRESSION_STMT);
    for (AstNode expression : expressions) {
      if (isAssignmentExpression(expression)) {
        getIdentifiersFromLongAssignmentExpression(allFields, expression, true);
      }
    }
  }

  public static void getIdentifiersFromLongAssignmentExpression(List<Token> fields, AstNode expression, boolean withSelf) {
    List<AstNode> assignedExpressions = expression.getChildren(PythonGrammar.TESTLIST_STAR_EXPR);
    assignedExpressions.remove(assignedExpressions.size() - 1);
    List<AstNode> tests = new LinkedList<>();
    for (AstNode assignedExpression : assignedExpressions){
      tests.addAll(assignedExpression.getDescendants(PythonGrammar.TEST));
    }
    for (AstNode test : tests) {
      if (withSelf){
        addSelfField(fields, test);
      } else {
        addSimpleField(fields, test);
      }
    }
  }

  private static void addSelfField(List<Token> fields, AstNode test) {
    if ("self".equals(test.getTokenValue())){
      AstNode trailer = test.getFirstDescendant(PythonGrammar.TRAILER);
      if (trailer != null && trailer.getFirstChild(PythonGrammar.NAME) != null){
        Token token = trailer.getFirstChild(PythonGrammar.NAME).getToken();
        if (!contains(fields, token)) {
          fields.add(token);
        }
      }
    }
  }

  private static void addSimpleField(List<Token> fields, AstNode test) {
    Token token = test.getToken();
    if (test.getNumberOfChildren() == 1 && test.getFirstChild().is(PythonGrammar.ATOM) && token.getType().equals(GenericTokenType.IDENTIFIER) && !contains(fields, token)){
      fields.add(token);
    }
  }

  public static boolean isAssignmentExpression(AstNode expression) {
    int numberOfChildren = expression.getNumberOfChildren();
    int numberOfAssign = expression.getChildren(PythonPunctuator.ASSIGN).size();
    if (numberOfChildren == 3 && numberOfAssign == 1){
      return true;
    }
    // a = b = c = 1
    return numberOfAssign > 0 && numberOfChildren % 2 == 1 && numberOfAssign * 2 + 1 == numberOfChildren;
  }

  public static boolean contains(List<Token> list, Token token) {
    for (Token currentToken : list) {
      if (currentToken.getValue().equals(token.getValue())) {
        return true;
      }
    }
    return false;
  }

  private static List<Token> findFieldsInClassBody(AstNode classDef) {
    List<Token> fields = new LinkedList<>();
    List<AstNode> statements = classDef.getFirstChild(PythonGrammar.SUITE).getChildren(PythonGrammar.STATEMENT);
    List<AstNode> expressions = new LinkedList<>();
    for (AstNode statement : statements){
      if (!statement.hasDescendant(PythonGrammar.FUNCDEF)){
        expressions.addAll(statement.getDescendants(PythonGrammar.EXPRESSION_STMT));
      }
    }
    for (AstNode expression : expressions) {
      if (isAssignmentExpression(expression)){
        getIdentifiersFromLongAssignmentExpression(fields, expression, false);
      }
    }
    return fields;
  }

  public static boolean classHasNoInheritance(AstNode classDef) {
    AstNode inheritanceClause = classDef.getFirstChild(PythonGrammar.ARGLIST);
    if (inheritanceClause == null){
      return true;
    } else {
      AstNode argument = inheritanceClause.getFirstChild(PythonGrammar.ARGUMENT);
      return argument != null && ("object".equals(argument.getTokenValue()));
    }
  }
}
