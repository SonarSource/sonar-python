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

import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.PyArgumentTree;
import org.sonar.python.api.tree.PyBinaryExpressionTree;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyQualifiedExpressionTree;
import org.sonar.python.api.tree.PyStringLiteralTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.checks.AbstractCallExpressionCheck2;
import org.sonar.python.semantic.Symbol;
import org.sonar.python.tree.BaseTreeVisitor;

@Rule(key = SQLQueriesCheck.CHECK_KEY)
public class SQLQueriesCheck extends AbstractCallExpressionCheck2 {
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
  public void initialize(Context context) {
    super.initialize(context);
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::visitFile);
  }

  private void visitFile(SubscriptionContext ctx) {
    isUsingDjangoModel = false;
    isUsingDjangoDBConnection = false;
    for (Symbol symbol : ctx.symbolTable().symbols(ctx.syntaxNode().astNode())) {
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
  public void visitNode(SubscriptionContext context) {
    PyCallExpressionTree callExpressionTree = (PyCallExpressionTree) context.syntaxNode();
    if(callExpressionTree.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
      // According to grammar definition `ATTRIBUTE_REF` has always at least one child of
      // kind NAME, hence we don't need to check for null on `getLastChild` call
      String functionName = ((PyQualifiedExpressionTree) callExpressionTree.callee()).name().name();
      if ((isSQLQueryFromDjangoModel(functionName) || isSQLQueryFromDjangoDBConnection(functionName)) && !isException(callExpressionTree, functionName)) {
        context.addIssue(callExpressionTree, MESSAGE);
      }
    }
    super.visitNode(context);
  }

  private boolean isException(PyCallExpressionTree callExpression, String functionName) {
    List<PyArgumentTree> argListNode = callExpression.arguments().arguments();
    if (extraContainsFormattedSqlQueries(argListNode, functionName)) {
      return false;
    }
    if (argListNode.isEmpty()) {
      return true;
    }
    PyArgumentTree arg = argListNode.get(0);
    return !isFormatted(arg.expression());
  }

  @Override
  protected boolean isException(PyCallExpressionTree callExpression) {
    return isException(callExpression, "");
  }

  private boolean isFormatted(PyExpressionTree tree) {
    FormattedStringVisitor visitor = new FormattedStringVisitor();
    tree.accept(visitor);
    return visitor.hasFormattedString;
  }

  private boolean extraContainsFormattedSqlQueries(List<PyArgumentTree> argListNode, String functionName) {
    if (functionName.equals("extra")) {
      return argListNode.stream()
        .filter(SQLQueriesCheck::isAssignment)
        .map(PyArgumentTree::expression)
        .anyMatch(this::isFormatted);
    }
    return false;
  }

  private static boolean isAssignment(PyArgumentTree arg) {
    return arg.equalToken() != null;
  }

  private static class FormattedStringVisitor extends BaseTreeVisitor {
    boolean hasFormattedString = false;

    @Override
    public void visitStringLiteral(PyStringLiteralTree pyStringLiteralTree) {
      super.visitStringLiteral(pyStringLiteralTree);
      hasFormattedString |= pyStringLiteralTree.stringElements().stream().anyMatch(se -> se.prefix().equalsIgnoreCase("f"));
    }

    @Override
    public void visitCallExpression(PyCallExpressionTree pyCallExpressionTree) {
      if(pyCallExpressionTree.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
        PyQualifiedExpressionTree callee = (PyQualifiedExpressionTree) pyCallExpressionTree.callee();
        hasFormattedString |=  callee.name().name().equals("format") && callee.qualifier().is(Tree.Kind.STRING_LITERAL);
      }
      super.visitCallExpression(pyCallExpressionTree);
    }

    @Override
    public void visitBinaryExpression(PyBinaryExpressionTree pyBinaryExpressionTree) {
      hasFormattedString |= pyBinaryExpressionTree.leftOperand().is(Tree.Kind.STRING_LITERAL) || pyBinaryExpressionTree.rightOperand().is(Tree.Kind.STRING_LITERAL);
      super.visitBinaryExpression(pyBinaryExpressionTree);
    }
  }
}
