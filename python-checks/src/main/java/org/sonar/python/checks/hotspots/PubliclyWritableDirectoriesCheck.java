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

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.api.tree.PyArgumentTree;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyQualifiedExpressionTree;
import org.sonar.python.api.tree.PyStringElementTree;
import org.sonar.python.api.tree.PyStringLiteralTree;
import org.sonar.python.api.tree.PySubscriptionExpressionTree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.semantic.Symbol;
import org.sonar.python.semantic.SymbolTable;

@Rule(key = "S5443")
public class PubliclyWritableDirectoriesCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Make sure publicly writable directories are used safely here.";
  private static final List<String> UNIX_WRITABLE_DIRECTORIES = Arrays.asList(
    "/tmp/", "/var/tmp/", "/usr/tmp/", "/dev/shm/", "/dev/mqueue/", "/run/lock/", "/var/run/lock/",
    "/library/caches/", "/users/shared/", "/private/tmp/", "/private/var/tmp/");
  private static final List<String> NONCOMPLIANT_ENVIRON_VARIABLES = Arrays.asList("tmpdir", "tmp");

  private static final Pattern WINDOWS_WRITABLE_DIRECTORIES = Pattern.compile("[^\\\\]*\\\\(Windows\\\\Temp|Temp|TMP)(\\\\.*|$)", Pattern.CASE_INSENSITIVE);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.STRING_ELEMENT, ctx -> {
      PyStringElementTree tree = (PyStringElementTree) ctx.syntaxNode();
      String stringElement = tree.trimmedQuotesValue().toLowerCase(Locale.ENGLISH);
      if (UNIX_WRITABLE_DIRECTORIES.stream().anyMatch(dir -> containsDirectory(stringElement, dir)) ||
        WINDOWS_WRITABLE_DIRECTORIES.matcher(stringElement).matches()) {
        ctx.addIssue(tree, MESSAGE);
      }
    });

    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, ctx -> {
      PyCallExpressionTree tree = (PyCallExpressionTree) ctx.syntaxNode();
      if (isOsEnvironGetter(tree.callee(), ctx.symbolTable()) &&
        tree.arguments().arguments().stream().map(PyArgumentTree::expression)
          .anyMatch(PubliclyWritableDirectoriesCheck::isNonCompliantOsEnvironArgument)) {
        ctx.addIssue(tree, MESSAGE);
      }
    });

    context.registerSyntaxNodeConsumer(Kind.SUBSCRIPTION, ctx -> {
      PySubscriptionExpressionTree tree = (PySubscriptionExpressionTree) ctx.syntaxNode();
      if (isOsEnvironQualifiedExpression(tree.object(), ctx.symbolTable()) && tree.subscripts().expressions().stream()
        .anyMatch(PubliclyWritableDirectoriesCheck::isNonCompliantOsEnvironArgument)) {
        ctx.addIssue(tree, MESSAGE);
      }
    });

  }

  private static boolean containsDirectory(String stringElement, String dir) {
    return stringElement.startsWith(dir) || stringElement.equals(dir.substring(0, dir.length() - 1));
  }

  private static boolean isNonCompliantOsEnvironArgument(PyExpressionTree expression) {
    return expression.is(Kind.STRING_LITERAL) &&
      ((PyStringLiteralTree) expression).stringElements().stream().map(s -> s.trimmedQuotesValue().toLowerCase(Locale.ENGLISH)).anyMatch(NONCOMPLIANT_ENVIRON_VARIABLES::contains);
  }

  // Could be either `os.environ.get` or `environ.get`
  private static boolean isOsEnvironGetter(PyExpressionTree expression, SymbolTable symbolTable) {
    if (!expression.is(Kind.QUALIFIED_EXPR)) {
      return false;
    }
    PyQualifiedExpressionTree qualifiedExpression = (PyQualifiedExpressionTree) expression;
    if (qualifiedExpression.name().name().equals("get")) {
      return isOsEnvironQualifiedExpression(qualifiedExpression.qualifier(), symbolTable);
    }
    return false;
  }

  // Could be either `os.environ` or `from os import environ; ... ; environ`
  private static boolean isOsEnvironQualifiedExpression(PyExpressionTree expression, SymbolTable symbolTable) {
    Symbol symbol = symbolTable.getSymbol(expression);
    if (symbol != null) {
      return symbol.qualifiedName().equals("os.environ");
    }
    if (!expression.is(Kind.QUALIFIED_EXPR)) {
      return false;
    }
    PyQualifiedExpressionTree qualifiedExpression = (PyQualifiedExpressionTree) expression;
    if (qualifiedExpression.qualifier().is(Kind.NAME)) {
      return ((PyNameTree) qualifiedExpression.qualifier()).name().equals("os") && qualifiedExpression.name().name().equals("environ");
    }
    return false;
  }
}
