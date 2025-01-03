/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks.hotspots;

import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.AbstractCallExpressionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;

@Rule(key = StandardInputCheck.CHECK_KEY)
public class StandardInputCheck extends AbstractCallExpressionCheck {
  public static final String CHECK_KEY = "S4829";
  private static final String MESSAGE = "Make sure that reading the standard input is safe here.";
  private static final Set<String> fileInputFunctions = immutableSet("fileinput.input", "fileinput.FileInput");
  private static final Set<String> sysFunctions = immutableSet("sys.stdin.read", "sys.stdin.readline", "sys.stdin.readlines",
    "sys.__stdin__.read", "sys.__stdin__.readline", "sys.__stdin__.readlines");
  private static final Set<String> questionableFunctionsBuiltIn = immutableSet(
    "raw_input", "input");
  private static final Set<String> questionablePropertyAccess = immutableSet("sys.stdin", "sys.__stdin__");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpr = (CallExpression) ctx.syntaxNode();
      if (questionableFunctionsBuiltIn.contains(getFunctionName(callExpr.callee()))) {
        ctx.addIssue(callExpr, message());
      } else {
        visitNode(ctx);
      }
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.NAME, ctx -> {
      Name name = (Name) ctx.syntaxNode();
      if (isWithinImport(name)) {
        return;
      }
      if (isQuestionablePropertyAccess(name)) {
        ctx.addIssue(name, message());
      }
    });
  }

  private static String getFunctionName(Expression expr) {
    String functionName = "";
    if (expr.is(Tree.Kind.NAME)) {
      functionName = ((Name) expr).name();
    }
    return functionName;
  }

  @Override
  protected boolean isException(CallExpression callExpression) {
    Symbol symbol = callExpression.calleeSymbol();
    return symbol != null && fileInputFunctions.contains(symbol.fullyQualifiedName()) && !callExpression.arguments().isEmpty();
  }

  private static boolean isQuestionablePropertyAccess(Name pyNameTree) {
    Tree parent = pyNameTree.parent();
    while (parent != null && !parent.is(Tree.Kind.CALL_EXPR)) {
      Tree grandParent = parent.parent();
      if (grandParent != null && grandParent.is(Tree.Kind.CALL_EXPR) && ((CallExpression) grandParent).callee() == parent) {
        // avoid raising twice the issue on call expressions like sys.stdin.read()
        return false;
      }
      parent = grandParent;
    }
    Symbol symbol = pyNameTree.symbol();
    return symbol != null && questionablePropertyAccess.contains(symbol.fullyQualifiedName());
  }

  @Override
  protected Set<String> functionsToCheck() {
    return Stream.concat(fileInputFunctions.stream(), sysFunctions.stream())
      .collect(Collectors.toSet());
  }

  @Override
  protected String message() {
    return MESSAGE;
  }
}
