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
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.python.IssueLocation;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.tree.PyArgumentTree;
import org.sonar.python.api.tree.PyBinaryExpressionTree;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.semantic.Symbol;

@Rule(key = RegexCheck.CHECK_KEY)
public class RegexCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S4784";
  private static final String MESSAGE = "Make sure that using a regular expression is safe here.";
  private static final int REGEX_ARGUMENT = 0;
  private static final Set<String> questionableFunctions = new HashSet<>(Arrays.asList(
    "django.core.validators.RegexValidator", "django.urls.re_path",
    "re.compile", "re.match", "re.search", "re.fullmatch", "re.split", "re.findall", "re.finditer", "re.sub", "re.subn",
    "regex.compile", "regex.match", "regex.search", "regex.fullmatch", "regex.split", "regex.findall", "regex.finditer", "regex.sub", "regex.subn",
    "regex.subf", "regex.subfn", "regex.splititer"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      PyCallExpressionTree call = (PyCallExpressionTree) ctx.syntaxNode();
      Symbol symbol = ctx.symbolTable().getSymbol(call.callee());
      if (symbol != null && questionableFunctions.contains(symbol.qualifiedName()) && !call.arguments().isEmpty()) {
        checkRegexArgument(call.arguments().get(REGEX_ARGUMENT), ctx);
      }
    });
  }

  private void checkRegexArgument(PyArgumentTree arg, SubscriptionContext ctx) {
    Symbol argSymbol = ctx.symbolTable().getSymbol(getExpression(arg.expression()));
    String literal = arg.firstToken().getValue();
    IssueLocation secondaryLocation = null;
    // TODO : this cannot be migrated to strongly typed AST as long as semantic is not migrated to strongly typed AST
    if (argSymbol != null && argSymbol.writeUsages().size() == 1) {
      AstNode expressionStatement = argSymbol.writeUsages().iterator().next().getFirstAncestor(PythonGrammar.EXPRESSION_STMT);
      if (isAssignment(expressionStatement)) {
        AstNode expression = expressionStatement.getChildren().get(2);
        literal = expression.getTokenValue();
        secondaryLocation = IssueLocation.preciseLocation(expression, "");
      }
    }
    if (isSuspiciousRegex(literal)) {
      PreciseIssue preciseIssue = ctx.addIssue(arg, MESSAGE);
      if (secondaryLocation != null) {
        preciseIssue.secondary(secondaryLocation);
      }
    }
  }

  private static PyExpressionTree getExpression(PyExpressionTree expr) {
    if (expr.is(Tree.Kind.MODULO) || expr.is(Tree.Kind.PLUS)) {
      return getExpression(((PyBinaryExpressionTree) expr).leftOperand());
    }
    return expr;
  }

  private static boolean isAssignment(@CheckForNull AstNode expressionStatement) {
    return expressionStatement != null &&
      expressionStatement.getChildren().size() == 3 &&
      expressionStatement.getChildren().get(1).is(PythonPunctuator.ASSIGN);
  }

  /**
   * This rule flags any execution of a hardcoded regular expression which has at least 3 characters and at least
   * two instances of any of the following characters: "*+{" (Example: (a+)*)
   */
  private static boolean isSuspiciousRegex(String regexp) {
    if (regexp.length() > 2) {
      int nOfSuspiciousChars = regexp.length() - regexp.replaceAll("[*+{]", "").length();
      return nOfSuspiciousChars > 1;
    }
    return false;
  }
}
