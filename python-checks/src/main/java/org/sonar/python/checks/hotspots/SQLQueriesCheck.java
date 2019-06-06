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
package org.sonar.python.checks.hotspots;

import com.sonar.sslr.api.AstNode;
import java.util.Collections;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.checks.AbstractCallExpressionCheck;
import org.sonar.python.semantic.Symbol;

@Rule(key = SQLQueriesCheck.CHECK_KEY)
public class SQLQueriesCheck extends AbstractCallExpressionCheck {
  public static final String CHECK_KEY = "S2077";
  private static final String MESSAGE = "Make sure that formatting this SQL query is safe here.";
  private boolean isUsingDjangoModel = false;
  private boolean isUsingDjangoDBConnection = false;

  @Override
  protected Set<String> functionsToCheck() {
    return Collections.singleton("django.db.models.expressions.RawSQL");
  }

  @Override
  protected String message() {
    return MESSAGE;
  }

  @Override
  public void visitFile(AstNode node) {
    isUsingDjangoModel = false;
    isUsingDjangoDBConnection = false;
    for (Symbol symbol : getContext().symbolTable().symbols(node)) {
      String qualifiedName = symbol.qualifiedName();
      if (qualifiedName.contains("django.db.models")) {
        isUsingDjangoModel = true;
      }
      if (qualifiedName.contains("django.db.connection")) {
        isUsingDjangoDBConnection = true;
      }
    }
  }

  private boolean isSQLQueryFromDjangoModel(String functionName) {
    return isUsingDjangoModel && (functionName.equals("raw") || functionName.equals("extra"));
  }

  private boolean isSQLQueryFromDjangoDBConnection(String functionName) {
    return isUsingDjangoDBConnection && functionName.equals("execute");
  }

  @Override
  public void visitNode(AstNode node) {
    AstNode attributeRef = node.getFirstChild(PythonGrammar.ATTRIBUTE_REF);
    if (attributeRef != null) {
      // According to grammar definition `ATTRIBUTE_REF` has always at least one child of
      // kind NAME, hence we don't need to check for null on `getLastChild` call
      String functionName = attributeRef.getLastChild(PythonGrammar.NAME).getTokenValue();
      if ((isSQLQueryFromDjangoModel(functionName) || isSQLQueryFromDjangoDBConnection(functionName)) && !isException(node, functionName)) {
        addIssue(node, MESSAGE);
      }
    }

    super.visitNode(node);
  }

  private boolean isException(AstNode callExpression, String functionName) {
    AstNode argListNode = callExpression.getFirstChild(PythonGrammar.ARGLIST);
    if (extraContainsFormattedSqlQueries(argListNode, functionName)) {
      return false;
    }
    if (argListNode != null) {
      AstNode sqlQueryNode = argListNode.getChildren().get(0);
      if (sqlQueryNode.getChildren().size() == 1 && sqlQueryNode.getFirstChild(PythonGrammar.TEST) != null) {
        AstNode testNode = sqlQueryNode.getFirstChild(PythonGrammar.TEST);
        return !isFormatted(testNode);
      }
    }
    return true;
  }

  @Override
  protected boolean isException(AstNode callExpression) {
    return isException(callExpression, "");
  }

  private boolean isFormatted(AstNode testNode) {
    if (testNode.getChildren().size() != 1) {
      return false;
    }
    return isStrFormatCall(testNode) ||
      isFormattedStringLiteral(testNode) ||
      testNode.getFirstChild(PythonGrammar.M_EXPR) != null ||
      testNode.getFirstChild(PythonGrammar.A_EXPR) != null;
  }

  private static boolean isStrFormatCall(AstNode testNode) {
    AstNode callExpr = testNode.getFirstChild(PythonGrammar.CALL_EXPR);
    if (callExpr != null) {
      AstNode attributeRef = callExpr.getFirstChild(PythonGrammar.ATTRIBUTE_REF);
      if (attributeRef != null) {
        return attributeRef.getFirstChild(PythonGrammar.ATOM).getFirstChild(PythonTokenType.STRING) != null &&
          attributeRef.getFirstChild(PythonGrammar.NAME).getTokenValue().equals("format");
      }
    }
    return false;
  }

  private static boolean isFormattedStringLiteral(AstNode testNode) {
    AstNode child = testNode.getFirstChild(PythonGrammar.ATOM);
    if (child != null) {
      AstNode string = child.getFirstChild(PythonTokenType.STRING);
      return string != null && (string.getTokenValue().startsWith("f")  || string.getTokenValue().startsWith("F"));
    }
    return false;
  }

  private boolean extraContainsFormattedSqlQueries(@CheckForNull AstNode argListNode, String functionName) {
    if (functionName.equals("extra")) {
      return argListNode != null && argListNode.getChildren(PythonGrammar.ARGUMENT).stream()
        .filter(SQLQueriesCheck::isAssignment)
        .map(argument -> argument.getChildren().get(2))
        .anyMatch(test -> test.getDescendants(PythonGrammar.TEST).stream().anyMatch(this::isFormatted));
    }
    return false;
  }

  private static boolean isAssignment(@CheckForNull AstNode node) {
    if (node == null || node.getChildren().size() != 3) {
      return false;
    }
    return node.getChildren().get(0).is(PythonGrammar.TEST) &&
      node.getChildren().get(1).is(PythonPunctuator.ASSIGN) &&
      node.getChildren().get(2).is(PythonGrammar.TEST);
  }
}
