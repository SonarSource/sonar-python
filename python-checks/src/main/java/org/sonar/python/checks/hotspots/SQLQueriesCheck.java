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
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.checks.AbstractCallExpressionCheck;
import org.sonar.python.semantic.Symbol;

@Rule(key = SQLQueriesCheck.CHECK_KEY)
public class SQLQueriesCheck extends AbstractCallExpressionCheck {
  public static final String CHECK_KEY = "S2077";
  private static final String MESSAGE = "Make sure that executing SQL queries is safe here.";
  private boolean isUsingDjangoModel = false;
  private boolean isUsingDjangoDBConnection = false;
  private boolean isUsingDjangoDBConnections = false;

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
    Set<String> symbols = getContext().symbolTable().symbols(node).stream()
      .map(Symbol::qualifiedName)
      .collect(Collectors.toSet());
    isUsingDjangoModel = symbols.contains("django.db.models");
    isUsingDjangoDBConnection = symbols.contains("django.db.connection");
    isUsingDjangoDBConnections = symbols.contains("django.db.connections");
  }

  private boolean isSQLQueryFromDjangoModel(String functionName) {
    return isUsingDjangoModel && functionName.equals("raw") || functionName.equals("extra");
  }

  private boolean isSQLQueryFromDjangoDBConnection(String functionName) {
    return (isUsingDjangoDBConnection || isUsingDjangoDBConnections) && functionName.equals("execute");
  }

  @Override
  public void visitNode(AstNode node) {
    AstNode attributeRef = node.getFirstChild(PythonGrammar.ATTRIBUTE_REF);
    if (attributeRef != null) {
      String functionName = attributeRef.getLastChild(PythonGrammar.NAME).getTokenValue();
      if (isSQLQueryFromDjangoModel(functionName) || isSQLQueryFromDjangoDBConnection(functionName)) {
        addIssue(node, MESSAGE);
      }
    }

    super.visitNode(node);
  }
}
