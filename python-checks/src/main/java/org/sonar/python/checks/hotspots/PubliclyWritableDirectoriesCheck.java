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

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.plugins.python.api.symbols.Symbol;

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
      StringElement tree = (StringElement) ctx.syntaxNode();
      String stringElement = Expressions.unescape(tree).toLowerCase(Locale.ENGLISH);
      if (UNIX_WRITABLE_DIRECTORIES.stream().anyMatch(dir -> containsDirectory(stringElement, dir)) ||
        WINDOWS_WRITABLE_DIRECTORIES.matcher(stringElement).matches()) {
        ctx.addIssue(tree, MESSAGE);
      }
    });

    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, ctx -> {
      CallExpression tree = (CallExpression) ctx.syntaxNode();
      List<Argument> arguments = tree.arguments();
      if (isOsEnvironGetter(tree) &&
        arguments.stream()
          .filter(arg -> arg.is(Kind.REGULAR_ARGUMENT))
          .map(RegularArgument.class::cast)
          .map(RegularArgument::expression)
          .anyMatch(PubliclyWritableDirectoriesCheck::isNonCompliantOsEnvironArgument)) {
        ctx.addIssue(tree, MESSAGE);
      }
    });

    context.registerSyntaxNodeConsumer(Kind.SUBSCRIPTION, ctx -> {
      SubscriptionExpression tree = (SubscriptionExpression) ctx.syntaxNode();
      if (isOsEnvironQualifiedExpression(tree.object()) && tree.subscripts().expressions().stream()
        .anyMatch(PubliclyWritableDirectoriesCheck::isNonCompliantOsEnvironArgument)) {
        ctx.addIssue(tree, MESSAGE);
      }
    });

  }

  private static boolean containsDirectory(String stringElement, String dir) {
    return stringElement.startsWith(dir) || stringElement.equals(dir.substring(0, dir.length() - 1));
  }

  private static boolean isNonCompliantOsEnvironArgument(Expression expression) {
    return expression.is(Kind.STRING_LITERAL) &&
      ((StringLiteral) expression).stringElements().stream().map(s -> Expressions.unescape(s).toLowerCase(Locale.ENGLISH)).anyMatch(NONCOMPLIANT_ENVIRON_VARIABLES::contains);
  }

  private static boolean isOsEnvironGetter(CallExpression callExpressionTree) {
    Symbol symbol = callExpressionTree.calleeSymbol();
    return symbol != null && "os.environ.get".equals(symbol.fullyQualifiedName());
  }

  private static boolean isOsEnvironQualifiedExpression(Expression expression) {
    if (expression instanceof HasSymbol hasSymbol) {
      Symbol symbol = hasSymbol.symbol();
      if (symbol != null) {
        return "os.environ".equals(symbol.fullyQualifiedName());
      }
    }
    return false;
  }
}
