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

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;

@Rule(key = "S2275")
public class StringFormatMisuseCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.MODULO, ctx -> {
      BinaryExpression expression = (BinaryExpression) ctx.syntaxNode();
      if (!expression.leftOperand().is(Tree.Kind.STRING_LITERAL)) {
        return;
      }

      StringLiteral formatString = ((StringLiteral) expression.leftOperand());
      Optional<StringFormat> formatOptional = StringFormat.createFromPrintfStyle(ctx, formatString, formatString.trimmedQuotesValue());

      if (!formatOptional.isPresent()) {
        // The string format contains invalid syntax.
        return;
      }

      StringFormat format = formatOptional.get();
      if (expression.rightOperand().is(Tree.Kind.TUPLE)) {
        checkTuples(ctx, format, ((Tuple) expression.rightOperand()));
      } else if (expression.rightOperand().is(Tree.Kind.DICTIONARY_LITERAL)) {
        checkDictionaries(ctx, format, ((DictionaryLiteral) expression.rightOperand()));
      } else if (format.numExpectedArguments() == 1) {
        format.replacementFields().get(0).validateArgument(expression.rightOperand());
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
      format.replacementFields().get(i).validateArgument(tuple.elements().get(i));
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

    Map<String, List<StringFormat.ReplacementField>> fieldMap = format.replacementFields().stream().collect(
      Collectors.groupingBy(StringFormat.ReplacementField::mappingKey));
    if (fieldMap.size() != dict.elements().size()) {
      reportInvalidArgumentSize(ctx, dict, fieldMap.size(), dict.elements().size());
      return;
    }

    for (DictionaryLiteralElement element : dict.elements()) {
      KeyValuePair pair = (KeyValuePair) element;
      String key = ((StringLiteral) pair.key()).trimmedQuotesValue();

      List<StringFormat.ReplacementField> fields = fieldMap.remove(key);
      if (fields == null) {
        // No such field
        continue;
      }

      fields.forEach(field -> field.validateArgument(pair.value()));
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

}
