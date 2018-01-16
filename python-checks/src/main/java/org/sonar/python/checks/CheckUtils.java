/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.python.checks;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Token;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;

public class CheckUtils {

  private static final Pattern STRING_INTERPOLATION_PREFIX = Pattern.compile("^[^'\"fF]*+[fF]");
  private static final Pattern STRING_LITERAL_QUOTE = Pattern.compile("[\"']");

  private CheckUtils() {

  }

  public static boolean isMethodDefinition(AstNode node) {
    if (!node.is(PythonGrammar.FUNCDEF)) {
      return false;
    }
    AstNode parent = node.getParent();
    for (int i = 0; i < 3; i++) {
      if (parent != null) {
        parent = parent.getParent();
      }
    }
    return parent != null && parent.is(PythonGrammar.CLASSDEF);
  }

  public static boolean isMethodOfNonDerivedClass(AstNode node) {
    return isMethodDefinition(node) && !classHasInheritance(node.getFirstAncestor(PythonGrammar.CLASSDEF));
  }


  public static boolean equalNodes(AstNode node1, AstNode node2) {
    if (!node1.getType().equals(node2.getType()) || node1.getNumberOfChildren() != node2.getNumberOfChildren()) {
      return false;
    }

    if (node1.getNumberOfChildren() == 0) {
      return node1.is(PythonTokenType.INDENT, PythonTokenType.DEDENT) || node1.getToken().getValue().equals(node2.getToken().getValue());
    }

    List<AstNode> children1 = node1.getChildren();
    List<AstNode> children2 = node2.getChildren();
    for (int i = 0; i < children1.size(); i++) {
      if (!equalNodes(children1.get(i), children2.get(i))) {
        return false;
      }
    }
    return true;
  }

  public static boolean insideFunction(AstNode astNode, AstNode funcDef) {
    return astNode.getFirstAncestor(PythonGrammar.FUNCDEF).equals(funcDef);
  }

  public static boolean classHasInheritance(AstNode classDef) {
    AstNode inheritanceClause = classDef.getFirstChild(PythonGrammar.ARGLIST);
    if (inheritanceClause == null) {
      return false;
    }
    List<AstNode> children = inheritanceClause.getChildren();
    if (children.isEmpty()) {
      return false;
    }
    return children.size() != 1 || !"object".equals(inheritanceClause.getFirstChild().getTokenValue());
  }

  public static boolean isAssignmentExpression(AstNode expression) {
    if (expression.is(PythonGrammar.EXPRESSION_STMT)) {
      AstNode assignNode = expression.getFirstChild(PythonGrammar.ANNASSIGN);
      if (assignNode != null && assignNode.getFirstChild(PythonPunctuator.ASSIGN) != null) {
        return true;
      }
    }
    int numberOfChildren = expression.getNumberOfChildren();
    int numberOfAssign = expression.getChildren(PythonPunctuator.ASSIGN).size();
    if (numberOfChildren == 3 && numberOfAssign == 1) {
      return true;
    }
    // a = b = c = 1
    return numberOfAssign > 0 && numberOfChildren % 2 != 0 && numberOfAssign * 2 + 1 == numberOfChildren;
  }

  public static boolean containsValue(List<Token> list, String value) {
    for (Token currentToken : list) {
      if (currentToken.getValue().equals(value)) {
        return true;
      }
    }
    return false;
  }

  public static boolean isStringInterpolation(Token token) {
    return token.getType().equals(PythonTokenType.STRING) &&
      STRING_INTERPOLATION_PREFIX.matcher(token.getOriginalValue()).find();
  }

  public static String stringLiteralContent(String stringLiteral) {
    Matcher quote = STRING_LITERAL_QUOTE.matcher(stringLiteral);
    if (!quote.find()) {
      throw new IllegalStateException("Invalid string literal: " + stringLiteral);
    }
    return stringLiteral.substring(quote.end(), stringLiteral.length() - 1);
  }

}
