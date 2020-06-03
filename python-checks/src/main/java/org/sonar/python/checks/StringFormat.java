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
import org.sonar.plugins.python.api.PythonCheck;
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
    "%" + "(?<field>(?:\\((?<mapkey>.*?)\\))?" + "(?<flags>[#\\-+0 ]*)?" + "(?<width>[0-9]*|\\*)?" +
      "(?:\\.(?<precision>[0-9]*|\\*))?" + "(?:[lLH])?" + "(?<type>[diueEfFgGoxXrsac]|%))?");

  private static final String PRINTF_NUMBER_CONVERTERS = "diueEfFgG";
  private static final String PRINTF_INTEGER_CONVERTERS = "oxX";

  public static Optional<StringFormat> createFromPrintfStyle(SubscriptionContext ctx, Expression lhsOperand, StringLiteral literal) {
    List<ReplacementField> result = new ArrayList<>();
    Matcher matcher = PRINTF_PARAMETER_PATTERN.matcher(literal.trimmedQuotesValue());

    while (matcher.find()) {
      if (matcher.group("field") == null) {
        // We matched a '%' sign, but could not match the rest of the field, the syntax is erroneous.
        reportSyntaxIssue(ctx, lhsOperand, literal, "Fix this formatted string's syntax.");
        return Optional.empty();
      }

      String mapKey = matcher.group("mapkey");
      String conversionType = matcher.group("type");

      if (conversionType.equals("%")) {
        // If the conversion type is '%', we are dealing with a '%%'
        continue;
      }

      String width = matcher.group("width");
      String precision = matcher.group("precision");
      if ("*".equals(width)) {
        result.add(new ReplacementField(printfWidthOrPrecisionValidator(ctx), null));
      }
      if ("*".equals(precision)) {
        result.add(new ReplacementField(printfWidthOrPrecisionValidator(ctx), null));
      }

      char conversionTypeChar = conversionType.charAt(0);
      result.add(new ReplacementField(printfConversionValidator(ctx, conversionTypeChar), mapKey));
    }

    StringFormat format = new StringFormat(result);
    if (format.hasPositionalFields() && format.hasNamedFields()) {
      reportSyntaxIssue(ctx, lhsOperand, literal, "Use only positional or only named field, don't mix them.");
      return Optional.empty();
    }

    return Optional.of(format);
  }

  private static void reportSyntaxIssue(SubscriptionContext ctx, Tree primary, Tree secondary, String message) {
    PythonCheck.PreciseIssue preciseIssue = ctx.addIssue(primary, message);
    if (primary != secondary) {
      preciseIssue.secondary(secondary, null);
    }
  }

  private static Consumer<Expression> printfWidthOrPrecisionValidator(SubscriptionContext ctx) {
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
    // The best would be to use 'InferredType::isCompatibleWith' against types created from protocols such as
    // 'SupportsFloat' and 'SupportsInt', but these are ambiguous symbols as they are defined both for Python 2 and 3.
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
