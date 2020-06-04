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
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

public class StringFormat {

  public abstract static class ReplacementField {
    private Consumer<Expression> validator;

    private ReplacementField(Consumer<Expression> validator) {
      this.validator = validator;
    }

    public abstract boolean isNamed();

    public abstract boolean isPositional();

    public abstract String name();

    public abstract int position();

    public void validateArgument(Expression expression) {
      this.validator.accept(expression);
    }
  }

  public static class NamedField extends ReplacementField {
    private String name;

    public NamedField(Consumer<Expression> validator, String name) {
      super(validator);
      this.name = name;
    }

    @Override
    public boolean isNamed() {
      return true;
    }

    @Override
    public boolean isPositional() {
      return false;
    }

    @Override
    public String name() {
      return this.name;
    }

    @Override
    public int position() {
      throw new NoSuchElementException();
    }
  }

  public static class PositionalField extends ReplacementField {
    private int position;

    public PositionalField(Consumer<Expression> validator, int position) {
      super(validator);
      this.position = position;
    }

    @Override
    public boolean isNamed() {
      return false;
    }

    @Override
    public boolean isPositional() {
      return true;
    }

    @Override
    public String name() {
      throw new NoSuchElementException();
    }

    @Override
    public int position() {
      return this.position;
    }
  }

  private List<ReplacementField> replacementFields;

  private StringFormat(List<ReplacementField> replacementFields) {
    this.replacementFields = replacementFields;
  }

  public List<ReplacementField> replacementFields() {
    return this.replacementFields;
  }

  public long numExpectedArguments() {
    long numPositional = this.replacementFields.stream()
      .filter(ReplacementField::isPositional)
      .map(ReplacementField::position)
      .distinct()
      .count();
    long numNamed = this.replacementFields.stream()
      .filter(ReplacementField::isNamed)
      .map(ReplacementField::name)
      .distinct()
      .count();

    return numPositional + numNamed;
  }

  public boolean hasPositionalFields() {
    return this.replacementFields.stream().anyMatch(ReplacementField::isPositional);
  }

  public boolean hasNamedFields() {
    return this.replacementFields.stream().anyMatch(ReplacementField::isNamed);
  }

  // See https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting
  private static final Pattern PRINTF_PARAMETER_PATTERN = Pattern.compile(
    "%" + "(?<field>(?:\\((?<mapkey>.*?)\\))?" + "(?<flags>[#\\-+0 ]*)?" + "(?<width>[0-9]*|\\*)?" +
      "(?:\\.(?<precision>[0-9]*|\\*))?" + "(?:[lLH])?" + "(?<type>[diueEfFgGoxXrsac]|%))?");

  private static final String PRINTF_NUMBER_CONVERTERS = "diueEfFgG";
  private static final String PRINTF_INTEGER_CONVERTERS = "oxX";

  private static final String FORMAT_FIELD_NAME_PATTERN = "(?<name>[^.\\[!:{}]+)?";


  // Format -> '{' [FieldName] ['!' Conversion] [':' FormatSpec] '}'
  // FormatSpec -> '{' [FieldName] '}'

  // Format -> '{' Field '}' | '{{' | '}}'
  // Field -> [Name] ('.' Name | '[' (Name | Number) ']')* [Flag] [':' Format ]
  // Flag -> '!' Character
  // See https://docs.python.org/3/library/string.html#formatstrings
  private static final Pattern FORMAT_PARAMETER_PATTERN = Pattern.compile(
    "\\{(?<field>\\{|(?:" + FORMAT_FIELD_NAME_PATTERN + "(?:(?:\\.[a-zA-Z0-9]+)|(?:\\[[a-zA-Z_0-9]+]))*(?:!(?<flag>[a-zA-Z]))?(:.*?)?}))?"
  );
  private static final String FORMAT_VALID_CONVERSION_FLAGS = "rsa";
  private static final Pattern FORMAT_NUMBER_PATTERN = Pattern.compile("^\\d+$");

  public static Optional<StringFormat> createFromStrFormatStyle(SubscriptionContext ctx, Tree tree, StringLiteral literal) {
    List<ReplacementField> result = new ArrayList<>();
    Matcher matcher = FORMAT_PARAMETER_PATTERN.matcher(literal.trimmedQuotesValue());

    int position = 0;
    boolean hasManualNumbering = false;
    while (matcher.find()) {
      String field = matcher.group("field");
      if (field == null) {
        // We matched a '{', but the rest of the field could not be matched.
        reportSyntaxIssue(ctx, tree, literal, "Fix this formatted string's syntax.");
        return Optional.empty();
      }

      if (field.equals("{")) {
        // We have a double '{{'.
        continue;
      }

      String name = matcher.group("name");
      String flag = matcher.group("flag");

      if (!checkFlag(flag, ctx, tree, literal)) {
        return Optional.empty();
      }

      if (name == null) {
        if (hasManualNumbering) {
          reportSyntaxIssue(ctx, tree, literal, "Use only manual or only automatic field numbering, don't mix them.");
          return Optional.empty();
        }
        result.add(new PositionalField(expression -> {}, position++));
      } else if (FORMAT_NUMBER_PATTERN.matcher(name).find()) {
        result.add(new PositionalField(expression -> {}, Integer.parseInt(name)));
        hasManualNumbering = true;
      } else {
        result.add(new NamedField(expression -> {}, name));
      }
    }

    StringFormat format = new StringFormat(result);

    return Optional.of(format);
  }

  private static boolean checkFlag(@Nullable String flag, SubscriptionContext ctx, Tree primary, Tree secondary) {
    if (flag == null) {
      return true;
    }

    char flagChar = flag.charAt(0);
    if (FORMAT_VALID_CONVERSION_FLAGS.indexOf(flagChar) == -1) {
      reportSyntaxIssue(ctx, primary, secondary, String.format("Fix this formatted string's syntax; !%c is not a valid conversion flag.", flagChar));
      return false;
    }

    return true;
  }

  public static Optional<StringFormat> createFromPrintfStyle(SubscriptionContext ctx, Expression lhsOperand, StringLiteral literal) {
    List<ReplacementField> result = new ArrayList<>();
    Matcher matcher = PRINTF_PARAMETER_PATTERN.matcher(literal.trimmedQuotesValue());

    int position = 0;
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
        result.add(new PositionalField(printfWidthOrPrecisionValidator(ctx), position++));
      }
      if ("*".equals(precision)) {
        result.add(new PositionalField(printfWidthOrPrecisionValidator(ctx), position++));
      }

      char conversionTypeChar = conversionType.charAt(0);
      if (mapKey != null) {
        result.add(new NamedField(printfConversionValidator(ctx, conversionTypeChar), mapKey));
      } else {
        result.add(new PositionalField(printfConversionValidator(ctx, conversionTypeChar), position++));
      }
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
