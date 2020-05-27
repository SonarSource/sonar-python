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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;

@Rule(key = "S2874")
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

      if (expression.rightOperand().is(Tree.Kind.TUPLE)) {
        checkTuples(ctx, format, ((Tuple) expression.rightOperand()));
      } else if (format.numExpectedArguments() == 1) {
        checkPrintfType(ctx, expression.rightOperand(), format.replacementFields().get(0));
      }
    });
  }

  private static void checkTuples(SubscriptionContext ctx, StringFormat format, Tuple tuple) {
    if (format.replacementFields().stream().anyMatch(field -> field.mappingKey() != null)) {
      ctx.addIssue(tuple, "Replace this formatting argument with a mapping.");
      return;
    }

    if (format.numExpectedArguments() > tuple.elements().size()) {
      ctx.addIssue(tuple, String.format("Add %d missing argument(s).", format.numExpectedArguments() - tuple.elements().size()));
      return;
    }

    if (format.numExpectedArguments() < tuple.elements().size()) {
      ctx.addIssue(tuple, String.format("Remove %d unexpected argument(s).", tuple.elements().size() - format.numExpectedArguments()));
      return;
    }

    for (int i = 0; i < tuple.elements().size(); ++i) {
      checkPrintfType(ctx, tuple.elements().get(i), format.replacementFields().get(i));
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
    } else if (conversionType == 'c' && cannotBeOfType(expression, "int") && !canBeSingleCharString(expression)) {
      ctx.addIssue(expression, String.format("Replace this value with an integer or a single character string as \"%%%c\" requires.", conversionType));
    }

    // No case for '%s', '%r' and '%a' - anything can be formatted with those.
  }

  private static boolean canBeSingleCharString(Expression expression) {
    if (!expression.type().canBeOrExtend("str")) {
      return false;
    }

    if (expression.is(Tree.Kind.STRING_LITERAL)) {
      return ((StringLiteral) expression).trimmedQuotesValue().length() == 1;
    }

    return true;
  }
}
