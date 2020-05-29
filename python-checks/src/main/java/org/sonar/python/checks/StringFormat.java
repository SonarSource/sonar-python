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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

public class StringFormat {

  public static class ReplacementField {
    private Consumer<Expression> validator;
    private String mappingKey;

    public ReplacementField(Consumer<Expression> validator, @Nullable String mappingKey) {
      this.validator = validator;
      this.mappingKey = mappingKey;
    }

    public void validateArgument(Expression expression) {
      this.validator.accept(expression);
    }

    @CheckForNull
    public String mappingKey() {
      return mappingKey;
    }
  }

  private List<ReplacementField> replacementFields;

  private StringFormat(List<ReplacementField> replacementFields) {
    this.replacementFields = replacementFields;
  }

  public List<ReplacementField> replacementFields() {
    return this.replacementFields;
  }

  public int numExpectedArguments() {
    return this.replacementFields.size();
  }

  public boolean hasPositionalFields() {
    return this.replacementFields.stream().anyMatch(field -> field.mappingKey() == null);
  }

  public boolean hasNamedFields() {
    return this.replacementFields.stream().anyMatch(field -> field.mappingKey() != null);
  }

  // See https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting
  private static final Pattern PRINTF_PARAMETER_PATTERN = Pattern.compile(
    "%" + "(?:\\((.*?)\\))?" + "([#\\-+0 ]*)?" + "([0-9]*|\\*)?" + "(?:\\.([0-9]*|\\*))?" + "([lLH])?" + "([A-Za-z]|%)");

  private static final String PRINTF_NUMBER_CONVERTERS = "diueEfFgG";
  private static final String PRINTF_INTEGER_CONVERTERS = "oxX";
  private static final String PRINTF_VALID_CONVERTERS = PRINTF_NUMBER_CONVERTERS + PRINTF_INTEGER_CONVERTERS + "rsac";

  public static Optional<StringFormat> createFromPrintfStyle(SubscriptionContext ctx, Tree tree, String input) {
    List<ReplacementField> result = new ArrayList<>();
    Matcher matcher = PRINTF_PARAMETER_PATTERN.matcher(input);

    while (matcher.find()) {
      String mapKey = matcher.group(1);
      String conversionType = matcher.group(6);

      if (conversionType.equals("%")) {
        // If the conversion type is '%', we are dealing with a '%%'
        continue;
      }

      char conversionTypeChar = conversionType.charAt(0);
      if (PRINTF_VALID_CONVERTERS.indexOf(conversionTypeChar) == -1) {
        ctx.addIssue(tree, String.format("Fix this formatted string's syntax; %%%c is not a valid conversion type.", conversionTypeChar));
        return Optional.empty();
      }

      String width = matcher.group(3);
      String precision = matcher.group(4);
      if ("*".equals(width)) {
        result.add(new ReplacementField(printfWidthOrPrecValidator(ctx), mapKey));
      }
      if ("*".equals(precision)) {
        result.add(new ReplacementField(printfWidthOrPrecValidator(ctx), mapKey));
      }

      result.add(new ReplacementField(printfConversionValidator(ctx, conversionTypeChar), mapKey));
    }

    if (input.contains("%") && result.isEmpty()) {
      // We consider the format erroneous if it contains '%' and could not match any replacement fields.
      ctx.addIssue(tree, "Fix this formatted string's syntax.");
      return Optional.empty();
    }

    StringFormat format = new StringFormat(result);
    if (format.hasPositionalFields() && format.hasNamedFields()) {
      ctx.addIssue(tree, "Use only positional or only named field, don't mix them.");
      return Optional.empty();
    }

    return Optional.of(new StringFormat(result));
  }

  private static Consumer<Expression> printfWidthOrPrecValidator(SubscriptionContext ctx) {
    return expression -> {
      if (cannotBeOfType(expression, "int")) {
        ctx.addIssue(expression, "Replace this value with an integer as \"*\" requires.");
      }
    };
  }

  private static Consumer<Expression> printfConversionValidator(SubscriptionContext ctx, char conversionType) {
    if (PRINTF_NUMBER_CONVERTERS.indexOf(conversionType) != -1) {
      return expr -> {
        if (cannotBeOfType(expr, "int", "float")) {
          ctx.addIssue(expr, String.format("Replace this value with a number as \"%%%c\" requires.", conversionType));
        }
      };
    }

    if (PRINTF_INTEGER_CONVERTERS.indexOf(conversionType) != -1) {
      return expr -> {
        if (cannotBeOfType(expr, "int")) {
          ctx.addIssue(expr, String.format("Replace this value with an integer as \"%%%c\" requires.", conversionType));
        }
      };
    }

    if (conversionType == 'c') {
      return expr -> {
        if (cannotBeOfType(expr, "int") && cannotBeSingleCharString(expr)) {
          ctx.addIssue(expr, String.format("Replace this value with an integer or a single character string as \"%%%c\" requires.", conversionType));
        }
      };
    }

    // No case for '%s', '%r' and '%a' - anything can be formatted with those.
    return expr -> {};
  }

  private static boolean cannotBeOfType(Expression expression, String... types) {
    return Arrays.stream(types).noneMatch(type -> expression.type().canBeOrExtend(type));
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
