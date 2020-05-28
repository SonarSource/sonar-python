/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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

import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;

@Rule(key = "S2275")
public class StringFormatMisuseCheck extends PythonSubscriptionCheck {

  private static final String PRINTF_NUMBER_CONVERTERS = "diueEfFgG";
  private static final String PRINTF_INTEGER_CONVERTERS = "oxX";
  private static final String PRINTF_VALID_CONVERTERS = PRINTF_NUMBER_CONVERTERS + PRINTF_INTEGER_CONVERTERS + "rsac";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.MODULO, ctx -> {
      BinaryExpression expression = (BinaryExpression) ctx.syntaxNode();
      if (!expression.leftOperand().is(Tree.Kind.STRING_LITERAL)) {
        return;
      }

      StringLiteral formatString = ((StringLiteral) expression.leftOperand());
      StringFormat format = StringFormat.createFromPrintfStyle(formatString.trimmedQuotesValue());

      // Check the format string's syntax
      for (StringFormat.ReplacementField field : format.replacementFields()) {
        char conversionType = field.conversionType().charAt(0);
        if (PRINTF_VALID_CONVERTERS.indexOf(conversionType) == -1) {
          ctx.addIssue(formatString, String.format("Fix this formatted string's syntax; %%%c is not a valid conversion type.", conversionType));
          return;
        }
      }

      if (format.hasPositionalFields() && format.hasNamedFields()) {
        ctx.addIssue(formatString, "Use only positional or only named field, don't mix them.");
        return;
      }

      if (expression.rightOperand().is(Tree.Kind.TUPLE)) {
        checkTuples(ctx, format, ((Tuple) expression.rightOperand()));
      } else if (expression.rightOperand().is(Tree.Kind.DICTIONARY_LITERAL)) {
        checkDictionaries(ctx, format, ((DictionaryLiteral) expression.rightOperand()));
      } else if (format.numExpectedArguments() == 1) {
        checkPrintfType(ctx, expression.rightOperand(), format.replacementFields().get(0));
      }
    });
  }

  private static void checkTuples(SubscriptionContext ctx, StringFormat format, Tuple tuple) {
    if (format.hasNamedFields()) {
      ctx.addIssue(tuple, "Replace this formatting argument with a mapping.");
      return;
    }

    if (format.numExpectedArguments() != tuple.elements().size()) {
      reportInvalidArgumentSize(ctx, tuple, format.numExpectedArguments(), tuple.elements().size());
      return;
    }

    for (int i = 0; i < tuple.elements().size(); ++i) {
      checkPrintfType(ctx, tuple.elements().get(i), format.replacementFields().get(i));
    }
  }

  private static void checkDictionaries(SubscriptionContext ctx, StringFormat format, DictionaryLiteral dict) {
    if (format.hasPositionalFields()) {
      ctx.addIssue(dict, "Replace this formatting argument with a tuple.");
      return;
    }

    if (dict.elements().stream().anyMatch(StringFormatMisuseCheck::isOutOfScopeDictionaryElement)) {
      // Do not bother with dictionaries containing unpacking expressions or keys which are not string literals.
      return;
    }

    if (format.numExpectedArguments() != dict.elements().size()) {
      reportInvalidArgumentSize(ctx, dict, format.numExpectedArguments(), dict.elements().size());
      return;
    }

    Map<String, StringFormat.ReplacementField> fieldMap = format.replacementFields().stream().collect(Collectors.toMap(
      StringFormat.ReplacementField::mappingKey, Function.identity()
    ));
    for (int i = 0; i < dict.elements().size(); ++i) {
      DictionaryLiteralElement element = dict.elements().get(i);
      KeyValuePair pair = (KeyValuePair) element;
      String key = ((StringLiteral) pair.key()).trimmedQuotesValue();

      StringFormat.ReplacementField field = fieldMap.remove(key);
      if (field == null) {
        // No such field
        continue;
      }

      checkPrintfType(ctx, pair.value(), field);
    }

    // Check if we have any unmatched field names left
    fieldMap.keySet().forEach(fieldName -> ctx.addIssue(dict, String.format("Provide a value for field \"%s\".", fieldName)));
  }

  private static boolean isOutOfScopeDictionaryElement(DictionaryLiteralElement element) {
    if (!element.is(Tree.Kind.KEY_VALUE_PAIR)) {
      return true;
    }

    KeyValuePair keyValuePair = (KeyValuePair) element;
    return !keyValuePair.key().is(Tree.Kind.STRING_LITERAL);
  }

  private static void reportInvalidArgumentSize(SubscriptionContext ctx, Tree tree, int expected, int actual) {
    if (expected > actual) {
      ctx.addIssue(tree, String.format("Add %d missing argument(s).", expected - actual));
    } else {
      ctx.addIssue(tree, String.format("Remove %d unexpected argument(s).", actual - expected));
    }
  }

  private static boolean cannotBeOfType(Expression expression, String... types) {
    return Arrays.stream(types).noneMatch(type -> expression.type().canBeOrExtend(type));
  }

  private static void checkPrintfType(SubscriptionContext ctx, Expression expression, StringFormat.ReplacementField field) {
    char conversionType = field.conversionType().charAt(0);
    if (PRINTF_NUMBER_CONVERTERS.indexOf(conversionType) != -1 && cannotBeOfType(expression, "int", "float")) {
      ctx.addIssue(expression, String.format("Replace this value with a number as \"%%%c\" requires.", conversionType));
    } else if (PRINTF_INTEGER_CONVERTERS.indexOf(conversionType) != -1 && cannotBeOfType(expression, "int")) {
      ctx.addIssue(expression, String.format("Replace this value with an integer as \"%%%c\" requires.", conversionType));
    } else if (conversionType == 'c' && cannotBeOfType(expression, "int") && cannotBeSingleCharString(expression)) {
      ctx.addIssue(expression, String.format("Replace this value with an integer or a single character string as \"%%%c\" requires.", conversionType));
    }

    // No case for '%s', '%r' and '%a' - anything can be formatted with those.
  }

  private static boolean cannotBeSingleCharString(Expression expression) {
    if (!expression.type().canBeOrExtend("str")) {
      return true;
    }

    if (expression.is(Tree.Kind.STRING_LITERAL)) {
      return ((StringLiteral) expression).trimmedQuotesValue().length() != 1;
    }

    return false;
  }
}
