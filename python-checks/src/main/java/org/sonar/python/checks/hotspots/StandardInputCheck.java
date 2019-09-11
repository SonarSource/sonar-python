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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyQualifiedExpressionTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.checks.AbstractCallExpressionCheck;
import org.sonar.python.semantic.Symbol;

@Rule(key = StandardInputCheck.CHECK_KEY)
public class StandardInputCheck extends AbstractCallExpressionCheck {
  public static final String CHECK_KEY = "S4829";
  private static final String MESSAGE = "Make sure that reading the standard input is safe here.";
  private static final Set<String> questionableFunctions = immutableSet("fileinput.input", "fileinput.FileInput");
  private static final Set<String> questionableFunctionsBuiltIn = immutableSet(
    "raw_input", "input", "sys.stdin.read", "sys.stdin.readline", "sys.stdin.readlines",
    "sys.__stdin__.read", "sys.__stdin__.readline", "sys.__stdin__.readlines");
  private static final Set<String> questionablePropertyAccess = immutableSet("sys.stdin", "sys.__stdin__");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      PyCallExpressionTree callExpr = (PyCallExpressionTree) ctx.syntaxNode();
      if (questionableFunctionsBuiltIn.contains(getFunctionName(callExpr.callee()))) {
        ctx.addIssue(callExpr, message());
      } else {
        visitNode(ctx);
      }
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.NAME, ctx -> {
      PyNameTree node = (PyNameTree) ctx.syntaxNode();
      if(isWithinImport(node)) {
        return;
      }
      if (isQuestionablePropertyAccess(node, ctx)) {
        ctx.addIssue(node, message());
      }
    });
  }

  private static String getFunctionName(PyExpressionTree expr) {
    String functionName = "";
    if (expr.is(Tree.Kind.QUALIFIED_EXPR)) {
      PyQualifiedExpressionTree qualExpr = (PyQualifiedExpressionTree) expr;
      functionName = getFunctionName(qualExpr.qualifier()) + "." + qualExpr.name().name();
    } else if (expr.is(Tree.Kind.NAME)) {
      functionName = ((PyNameTree) expr).name();
    }
    return functionName;
  }

  @Override
  protected boolean isException(PyCallExpressionTree callExpression) {
    return !callExpression.arguments().isEmpty();
  }

  private boolean isQuestionablePropertyAccess(PyNameTree pyNameTree, SubscriptionContext ctx) {
    Symbol symbol = ctx.symbolTable().getSymbol(pyNameTree);
    return symbol != null && questionablePropertyAccess.contains(symbol.qualifiedName());
  }

  @Override
  protected Set<String> functionsToCheck() {
    return questionableFunctions;
  }

  @Override
  protected String message() {
    return MESSAGE;
  }
}
