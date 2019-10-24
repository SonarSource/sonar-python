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

import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.StarredExpression;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.checks.Expressions;
import org.sonar.python.semantic.Symbol;

@Rule(key = "S5439")
public class DisabledHtmlAutoEscapeCheck extends PythonSubscriptionCheck {

  private static final String AUTO_ESCAPE = "autoescape";
  private static final String MESSAGE = "Remove this configuration disabling autoescape globally.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, ctx ->
      checkCallExpression(ctx, (CallExpression) ctx.syntaxNode())
    );
    context.registerSyntaxNodeConsumer(Kind.KEY_VALUE_PAIR, ctx ->
      checkKeyValuePair(ctx, (KeyValuePair) ctx.syntaxNode())
    );
  }

  private static void checkKeyValuePair(SubscriptionContext ctx, KeyValuePair keyValue) {
    if (!"settings.py".equals(ctx.pythonFile().fileName())) {
      return;
    }
    if (isStringLiteral(keyValue.key(), AUTO_ESCAPE) && Expressions.isFalsy(keyValue.value())) {
      ctx.addIssue(keyValue, MESSAGE);
    }
  }

  private static boolean isStringLiteral(Expression tree, String testedValue) {
    return tree.is(Kind.STRING_LITERAL) && testedValue.equals(((StringLiteral) tree).trimmedQuotesValue());
  }

  private static void checkCallExpression(SubscriptionContext ctx, CallExpression call) {
    Symbol symbol = call.calleeSymbol();

    if (symbol != null && "jinja2.Environment".equals(symbol.fullyQualifiedName())) {
      List<Argument> arguments = call.arguments();

      for (Argument argument : arguments) {
        if (argument.is(Kind.STARRED_EXPR)) {
          Expression expression = ((StarredExpression) argument).expression();
          if (expression.is(Kind.NAME)) {
            checkJinjaOptions(ctx, call, (Name) expression);
            return;
          }
        }
      }

      Stream<RegularArgument> autoEscapeArgs = arguments.stream()
        .filter(arg -> arg.is(Kind.REGULAR_ARGUMENT))
        .map(RegularArgument.class::cast)
        .filter(DisabledHtmlAutoEscapeCheck::isAutoEscapeArgument);
      if (autoEscapeArgs.allMatch(arg -> Expressions.isFalsy(arg.expression()))) {
        ctx.addIssue(call, MESSAGE);
      }
    }
  }

  private static void checkJinjaOptions(SubscriptionContext ctx, CallExpression call, Name expression) {
    Expression options = Expressions.singleAssignedValue(expression);
    if (options != null && options.is(Kind.DICTIONARY_LITERAL)) {
      DictionaryLiteral dict = (DictionaryLiteral) options;
      Optional<Expression> autoEscapeOption = dict.elements().stream()
        .filter(elem -> elem.is(Kind.KEY_VALUE_PAIR))
        .map(KeyValuePair.class::cast)
        .filter(kv -> isStringLiteral(kv.key(), AUTO_ESCAPE))
        .map(KeyValuePair::value)
        .findFirst();
      if (!autoEscapeOption.isPresent() || Expressions.isFalsy(autoEscapeOption.get())) {
        ctx.addIssue(call, MESSAGE);
      }
    }
  }

  private static boolean isAutoEscapeArgument(RegularArgument argument) {
    Name keyword = argument.keywordArgument();
    return keyword != null && AUTO_ESCAPE.equals(keyword.name());
  }
}
