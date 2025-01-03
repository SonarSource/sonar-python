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
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

public class StringFormat {

  private static final String SYNTAX_ERROR_MESSAGE = "Fix this formatted string's syntax.";

  private static final BiConsumer<SubscriptionContext, Expression> DO_NOTHING_VALIDATOR = (ctx, expr) -> {
  };

  // See https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting
  private static final Pattern PRINTF_PARAMETER_PATTERN = Pattern.compile(
    "%" + "(?<field>(?:\\((?<mapkey>.*?)\\))?" + "(?<flags>[#\\-+0 ]*)?" + "(?<width>[0-9]*|\\*)?" +
      "(?:\\.(?<precision>[0-9]*|\\*))?" + "(?:[lLH])?" + "(?<type>[diueEfFgGoxXrsac]|%))?");

  private static final Pattern FORMAT_FIELD_PATTERN = Pattern.compile("^(?<name>[^.\\[!:{}]+)?(?:(?:\\.[a-zA-Z0-9_]+)|(?:\\[[^]]+]))*");
  private static final Pattern FORMAT_NUMBER_PATTERN = Pattern.compile("^\\d+$");
  private static final Pattern FORMAT_UNICODE_PATTERN = Pattern.compile("^\\\\N\\{[a-zA-Z0-9-_\\s]*}");

  private static final String FORMAT_VALID_CONVERSION_FLAGS = "rsa";

  private static final String PRINTF_NUMBER_CONVERTERS = "diueEfFgG";
  private static final String PRINTF_INTEGER_CONVERTERS = "oxX";

  /**
   * Represents a named or positional replacement field inside a format string.
   */
  public abstract static class ReplacementField {
    private BiConsumer<SubscriptionContext, Expression> validator;

    private ReplacementField(BiConsumer<SubscriptionContext, Expression> validator) {
      this.validator = validator;
    }

    public abstract boolean isNamed();

    public abstract boolean isPositional();

    public abstract String name();

    public abstract int position();

    public void validateArgument(SubscriptionContext ctx, Expression expression) {
      this.validator.accept(ctx, expression);
    }
  }

  public static class NamedField extends ReplacementField {
    private String name;

    public NamedField(BiConsumer<SubscriptionContext, Expression> validator, String name) {
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

    public PositionalField(BiConsumer<SubscriptionContext, Expression> validator, int position) {
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

  public long numExpectedPositional() {
    return this.replacementFields.stream()
      .filter(ReplacementField::isPositional)
      .map(ReplacementField::position)
      .distinct()
      .count();
  }

  public long numExpectedArguments() {
    long numNamed = this.replacementFields.stream()
      .filter(ReplacementField::isNamed)
      .map(ReplacementField::name)
      .distinct()
      .count();

    return this.numExpectedPositional() + numNamed;
  }

  public boolean hasPositionalFields() {
    return this.replacementFields.stream().anyMatch(ReplacementField::isPositional);
  }

  public boolean hasNamedFields() {
    return this.replacementFields.stream().anyMatch(ReplacementField::isNamed);
  }

  private enum ParseState {
    INIT, RCURLY
  }
  private static class StrFormatParser {
    private boolean hasManualNumbering = false;
    private boolean hasAutoNumbering = false;
    private int autoNumberingPos = 0;

    private ParseState state = ParseState.INIT;
    private int nesting = 0;
    private List<ReplacementField> result;

    private Consumer<String> issueReporter;
    private String value;
    private int pos = 0;

    public StrFormatParser(Consumer<String> issueReporter, String value) {
      this.issueReporter = issueReporter;
      this.value = value;
      this.pos = 0;
    }

    public Optional<StringFormat> parse() {
      pos = 0;
      result = new ArrayList<>();

      while (pos < value.length()) {
        char current = value.charAt(pos);
        if (state == ParseState.INIT) {
          if (!tryParsingInitial(current)) {
            return Optional.empty();
          }
        } else if (state == ParseState.RCURLY) {
          if (current != '}') {
            issueReporter.accept(SYNTAX_ERROR_MESSAGE);
            return Optional.empty();
          }
          state = ParseState.INIT;
        }

        pos += 1;
      }

      if (!checkParserState()) {
        // The parser has reached the end of the string in a invalid state.
        return Optional.empty();
      }

      return Optional.of(new StringFormat(result));
    }

    private boolean checkParserState() {
      if (nesting != 0 || state != ParseState.INIT) {
        issueReporter.accept(SYNTAX_ERROR_MESSAGE);
        return false;
      }

      if (hasManualNumbering && hasAutoNumbering) {
        issueReporter.accept("Use only manual or only automatic field numbering, don't mix them.");
        return false;
      }

      return true;
    }

    private boolean tryParsingInitial(char current) {
      if (current == '{') {
        return tryParsingField();
      } else if (current == '}') {
        state = ParseState.RCURLY;
      } else if (current == '\\') {
        // See if this is a unicode escape
        Matcher unicodeMatcher = FORMAT_UNICODE_PATTERN.matcher(this.value).region(pos, this.value.length());
        if (unicodeMatcher.find()) {
          pos = unicodeMatcher.end() - 1;
        }
      }

      return true;
    }

    private boolean tryParsingField() {
      FieldParser fieldParser = new FieldParser(this, value.substring(pos), 0);
      boolean successful = fieldParser.tryParse();
      this.pos += fieldParser.getPos() - 1;
      return successful;
    }

    public void reportIssue(String issue) {
      issueReporter.accept(issue);
    }

    public void addField(@Nullable String name) {
      result.add(createField(name));
    }

    private ReplacementField createField(@Nullable String name) {
      if (name == null) {
        hasAutoNumbering = true;
        int currentPos = autoNumberingPos;
        autoNumberingPos++;
        return new PositionalField(DO_NOTHING_VALIDATOR, currentPos);
      } else if (FORMAT_NUMBER_PATTERN.matcher(name).find()) {
        hasManualNumbering = true;
        return new PositionalField(DO_NOTHING_VALIDATOR, Integer.parseInt(name));
      } else {
        return new NamedField(DO_NOTHING_VALIDATOR, name);
      }
    }

  }

  private enum FieldParseState {
    LCURLY, FIELD, FLAG, FLAG_CHARACTER, FORMAT, FINISHED
  }
  private static class FieldParser {
    private StrFormatParser parent;

    private String currentFieldName = null;
    private FieldParseState state = FieldParseState.LCURLY;
    private int nesting;

    private String value;
    private int pos;
    private Matcher fieldContentMatcher;

    public FieldParser(StrFormatParser parent, String value, int nesting) {
      this.parent = parent;
      this.value = value;
      this.pos = 1;
      this.fieldContentMatcher = FORMAT_FIELD_PATTERN.matcher(this.value);
      this.nesting = nesting;
    }

    public int getPos() {
      return pos;
    }

    public boolean tryParse() {
      pos = 1;

      while (pos < value.length() && state != FieldParseState.FINISHED) {
        char current = value.charAt(pos);
        boolean successful = switch (state) {
          case LCURLY -> {
            pos = parseFieldName(current, pos);
            yield true;
          }
          case FIELD -> tryParseField(current);
          case FLAG -> tryParseFlag(current);
          case FLAG_CHARACTER -> tryParseFlagCharacter(current);
          case FORMAT -> tryParseFormatSpecifier(current);
          case FINISHED -> throw new IllegalStateException("Unexpected value: " + state);
        };

        if (!successful) {
          return false;
        }

        pos += 1;
      }

      return checkParserState();
    }

    private boolean checkParserState() {
      if (state != FieldParseState.FINISHED) {
        parent.reportIssue(SYNTAX_ERROR_MESSAGE);
        return false;
      }
      return true;
    }

    private boolean tryParseFormatSpecifier(char current) {
      if (current == '{') {
        if (!tryParsingNestedField()) {
          return false;
        }
      } else if (current == '}') {
        addCurrentField();
        state = FieldParseState.FINISHED;
      }
      return true;
    }

    private boolean tryParsingNestedField() {
      if (this.nesting > 0) {
        parent.reportIssue("Fix this formatted string's syntax; Deep nesting is not allowed.");
        return false;
      }
      FieldParser fieldParser = new FieldParser(parent, value.substring(pos), this.nesting + 1);

      boolean successful = fieldParser.tryParse();
      this.pos += fieldParser.getPos() - 1;
      return successful;
    }

    private boolean tryParseFlagCharacter(char current) {
      if (current == ':') {
        state = FieldParseState.FORMAT;
      } else if (current == '}') {
        addCurrentField();
        state = FieldParseState.FINISHED;
      } else {
        parent.reportIssue(SYNTAX_ERROR_MESSAGE);
        return false;
      }

      return true;
    }

    private boolean tryParseField(char current) {
      if (current == '!') {
        state = FieldParseState.FLAG;
      } else if (current == ':') {
        state = FieldParseState.FORMAT;
      } else if (current == '}') {
        addCurrentField();
        state = FieldParseState.FINISHED;
      } else {
        parent.reportIssue(SYNTAX_ERROR_MESSAGE);
        return false;
      }
      return true;
    }

    private boolean tryParseFlag(char current) {
      if (FORMAT_VALID_CONVERSION_FLAGS.indexOf(current) == -1) {
        parent.reportIssue(String.format("Fix this formatted string's syntax; !%c is not a valid conversion flag.", current));
        return false;
      }
      state = FieldParseState.FLAG_CHARACTER;
      return true;
    }

    private int parseFieldName(char current, int pos) {
      if (current == '{') {
        state = FieldParseState.FINISHED;
      } else {
        state = FieldParseState.FIELD;
        if (fieldContentMatcher.region(pos, value.length()).find()) {
          // This should always match (if nothing else, an empty string), but be defensive
          currentFieldName = fieldContentMatcher.group("name");
          pos = fieldContentMatcher.end() - 1;
        }
      }

      return pos;
    }

    private void addCurrentField() {
      parent.addField(currentFieldName);
    }
  }

  public static Optional<StringFormat> createFromStrFormatStyle(Consumer<String> issueReporter, String value) {
    // Format -> '{' [FieldName] ['!' Conversion] [':' FormatSpec*] '}'
    // FormatSpec -> '{' [FieldName] '}' | Character
    // FieldName -> [Name] ('.' Name | '[' (Name | Number) ']')*
    // See https://docs.python.org/3/library/string.html#formatstrings
    return new StrFormatParser(issueReporter, value).parse();
  }

  public static Optional<StringFormat> createFromPrintfStyle(Consumer<String> issueReporter, String value) {
    List<ReplacementField> result = new ArrayList<>();
    Matcher matcher = PRINTF_PARAMETER_PATTERN.matcher(value);

    int position = 0;
    while (matcher.find()) {
      if (matcher.group("field") == null) {
        // We matched a '%' sign, but could not match the rest of the field, the syntax is erroneous.
        issueReporter.accept(SYNTAX_ERROR_MESSAGE);
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
        int currentPos = position;
        position++;
        result.add(new PositionalField(printfWidthOrPrecisionValidator(), currentPos));
      }
      if ("*".equals(precision)) {
        int currentPos = position;
        position++;
        result.add(new PositionalField(printfWidthOrPrecisionValidator(), currentPos));
      }

      char conversionTypeChar = conversionType.charAt(0);
      if (mapKey != null) {
        result.add(new NamedField(printfConversionValidator(conversionTypeChar), mapKey));
      } else {
        int currentPos = position;
        position++;
        result.add(new PositionalField(printfConversionValidator(conversionTypeChar), currentPos));
      }
    }

    StringFormat format = new StringFormat(result);
    if (format.hasPositionalFields() && format.hasNamedFields()) {
      issueReporter.accept("Use only positional or only named fields, don't mix them.");
      return Optional.empty();
    }

    return Optional.of(format);
  }

  private static BiConsumer<SubscriptionContext, Expression> printfWidthOrPrecisionValidator() {
    return (ctx, expression) -> {
      if (cannotBeOfType(expression, "int")) {
        ctx.addIssue(expression, "Replace this value with an integer as \"*\" requires.");
      }
    };
  }

  private static BiConsumer<SubscriptionContext, Expression> printfConversionValidator(char conversionType) {
    if (PRINTF_NUMBER_CONVERTERS.indexOf(conversionType) != -1) {
      return (ctx, expression) -> {
        if (cannotBeOfType(expression, "int", "float")) {
          ctx.addIssue(expression, String.format("Replace this value with a number as \"%%%c\" requires.", conversionType));
        }
      };
    }

    if (PRINTF_INTEGER_CONVERTERS.indexOf(conversionType) != -1) {
      return (ctx, expression) -> {
        if (cannotBeOfType(expression, "int")) {
          ctx.addIssue(expression, String.format("Replace this value with an integer as \"%%%c\" requires.", conversionType));
        }
      };
    }

    if (conversionType == 'c') {
      return (ctx, expression) -> {
        if (cannotBeOfType(expression, "int") && cannotBeSingleCharString(expression)) {
          ctx.addIssue(expression, String.format("Replace this value with an integer or a single character string as \"%%%c\" requires.", conversionType));
        }
      };
    }

    // No case for '%s', '%r' and '%a' - anything can be formatted with those.
    return (ctx, expression) -> {
    };
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
