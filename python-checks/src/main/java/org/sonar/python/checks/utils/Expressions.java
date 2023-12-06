/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.checks.utils;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.tree.TreeUtils;

public class Expressions {

  private static final Set<String> ZERO_VALUES = new HashSet<>(Arrays.asList("0", "0.0", "0j"));

  private Expressions() {
  }

  // https://docs.python.org/3/library/stdtypes.html#truth-value-testing
  public static boolean isFalsy(@Nullable Expression expression) {
    if (expression == null) {
      return false;
    }
    switch (expression.getKind()) {
      case NAME: 
        return "False".equals(((Name) expression).name());
      case NONE:
        return true;
      case STRING_LITERAL:
        return unescape((StringLiteral) expression).isEmpty();
      case NUMERIC_LITERAL:
        return ZERO_VALUES.contains(((NumericLiteral) expression).valueAsString());
      case LIST_LITERAL:
        return ((ListLiteral) expression).elements().expressions().isEmpty();
      case TUPLE:
        return ((Tuple) expression).elements().isEmpty();
      case DICTIONARY_LITERAL:
        return ((DictionaryLiteral) expression).elements().isEmpty();
      default:
        return false;
    }
  }

  // https://docs.python.org/3/library/stdtypes.html#truth-value-testing
  public static boolean isTruthy(@Nullable Expression expression) {
    if (expression == null) {
      return false;
    }
    switch (expression.getKind()) {
      case NAME:
        return "True".equals(((Name) expression).name());
      case STRING_LITERAL:
      case NUMERIC_LITERAL:
      case LIST_LITERAL:
      case TUPLE:
      case SET_LITERAL:
      case DICTIONARY_LITERAL:
        return !isFalsy(expression);
      default:
        return false;
    }
  }

  public static Expression singleAssignedValue(Name name) {
    return singleAssignedValue(name, new HashSet<>());
  }

  public static Expression singleAssignedValue(Name name, Set<Name> visited) {
    if (visited.contains(name)) {
      return null;
    }
    visited.add(name);
    Symbol symbol = name.symbol();
    if (symbol == null) {
      return null;
    }
    Expression result = null;
    for (Usage usage : symbol.usages()) {
      if (usage.kind() == Usage.Kind.ASSIGNMENT_LHS) {
        if (result != null) {
          return null;
        }
        Tree parent = usage.tree().parent();
        if (parent.is(Kind.EXPRESSION_LIST) &&
          ((ExpressionList) parent).expressions().size() == 1 &&
          parent.parent().is(Kind.ASSIGNMENT_STMT)) {

          result = ((AssignmentStatement) parent.parent()).assignedValue();
        } else {
          return null;
        }
      } else if (usage.isBindingUsage()) {
        return null;
      }
    }
    return result;
  }

  public static Expression singleAssignedNonNameValue(Name name) {
    Set<Name> visited = new HashSet<>();
    Expression result = singleAssignedValue(name, visited);
    while (result != null && result.is(Kind.NAME)) {
      result = singleAssignedValue((Name) result, visited);
    }
    return result;
  }

  public static Expression removeParentheses(@Nullable Expression expression) {
    if (expression == null) {
      return null;
    }
    Expression res = expression;
    while (res.is(Kind.PARENTHESIZED)) {
      res = ((ParenthesizedExpression) res).expression();
    }
    return res;
  }

  public static Expression ifNameGetSingleAssignedNonNameValue(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return Expressions.singleAssignedNonNameValue((Name) expression);
    }
    return expression;
  }

  public static Optional<List<Expression>> expressionsFromListOrTuple(Expression expression) {
    return TreeUtils.toOptionalInstanceOf(ListLiteral.class, expression)
      .map(ListLiteral::elements)
      .map(ExpressionList::expressions)
      .or(() -> TreeUtils.toOptionalInstanceOf(Tuple.class, expression)
        .map(Tuple::elements));
  }

  /**
   * @return concatenation of all underlying StringElement text values without quotes and with escape sequences replacement.
   * @see <a href="https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals">2.4.1. String and Bytes literals</a>
   */
  public static String unescape(StringLiteral stringLiteral) {
    List<StringElement> elements = stringLiteral.stringElements();
    if (elements.size() == 1) {
      return unescape(elements.get(0));
    }
    return elements.stream()
      .map(Expressions::unescape)
      .collect(Collectors.joining());
  }

  /**
   * @return the string content of the given StringElement text values without quotes and with escape sequences replacement.
   * @see <a href="https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals">2.4.1. String and Bytes literals</a>
   */
  public static String unescape(StringElement stringElement) {
    String lowerCasePrefix = stringElement.prefix().toLowerCase(Locale.ROOT);
    String valueWithoutQuotes = stringElement.trimmedQuotesValue();
    boolean isEscaped = lowerCasePrefix.indexOf('r') == -1;
    boolean isBytesLiteral = lowerCasePrefix.indexOf('b') != -1;
    if (isEscaped) {
      valueWithoutQuotes = unescapeString(valueWithoutQuotes, isBytesLiteral);
    }
    return valueWithoutQuotes;
  }

  /**
   * @param value to unescape according to python string and bytes literals conventions
   * @param isBytesLiteral knowing if it's a string, or an array of bytes is important because
   *                       python string uses 16 bits characters and support backslash u and U escape sequences
   *                       <code>'\u0061' == 'a'</code>
   *                       python bytes array uses 8 bits values and does not support and unescape backslash u and U escape sequences
   *                       <code>b'\u0061' != b'a'</code>
   *                       <code>b'\u0061' == b'\\u0061'</code>
   * @return unescaped value
   */
  // Visible for testing
  public static String unescapeString(String value, boolean isBytesLiteral) {
    if (value.indexOf('\\') == -1) {
      return value;
    }
    int length = value.length();
    StringBuilder sb = new StringBuilder(length);
    int i = 0;
    while (i < length) {
      char ch = value.charAt(i);
      if (ch != '\\') {
        sb.append(ch);
        i++;
      } else {
        EscapeSequence escapeSequence = EscapeSequence.extract(value, i, isBytesLiteral);
        sb.append(escapeSequence.unescapedValue);
        i += escapeSequence.escapedLength;
      }
    }
    return sb.toString();
  }

  private static class EscapeSequence {

    private static final int HEXADECIMAL_RADIX = 16;

    private static final EscapeSequence IGNORE = new EscapeSequence(1, "\\");

    private static final char[] UNESCAPED_CHAR = new char['v' + 1];
    static {
      UNESCAPED_CHAR['\\'] = '\\';
      UNESCAPED_CHAR['\''] = '\'';
      UNESCAPED_CHAR['\"'] = '\"';
      UNESCAPED_CHAR['a'] = '\u0007';
      UNESCAPED_CHAR['b'] = '\b';
      UNESCAPED_CHAR['f'] = '\f';
      UNESCAPED_CHAR['n'] = '\n';
      UNESCAPED_CHAR['r'] = '\r';
      UNESCAPED_CHAR['t'] = '\t';
      UNESCAPED_CHAR['v'] = '\u000b';
    }

    private final int escapedLength;
    private final String unescapedValue;

    private EscapeSequence(int escapedLength, String unescapedValue) {
      this.escapedLength = escapedLength;
      this.unescapedValue = unescapedValue;
    }

    private static EscapeSequence extract(String value, int i, boolean isBytesLiteral) {
      if (i == value.length() - 1) {
        return IGNORE;
      }
      char nextChar = value.charAt(i + 1);
      char unescaped = nextChar < UNESCAPED_CHAR.length ? UNESCAPED_CHAR[nextChar] : '\0';
      if (unescaped != '\0') {
        return new EscapeSequence(2, String.valueOf(unescaped));
      } else if (nextChar == '\n') {
        // ignored line break (linux end of line)
        return new EscapeSequence(2, "");
      } else if (nextChar == '\r') {
        // ignored line break (windows and mac end of line)
        return new EscapeSequence(i + 2 < value.length() && value.charAt(i + 2) == '\n' ? 3 : 2, "");
      } else if (nextChar == 'x') {
        return extractHexadecimal(value, i, 2);
      } else if (nextChar == 'u' && !isBytesLiteral) {
        return extractHexadecimal(value, i, 4);
      } else if (nextChar == 'U' && !isBytesLiteral) {
        return extractHexadecimal(value, i, 8);
      } else if (nextChar == 'N') {
        // escape sequence by unicode name is not supported, require java 9 to benefit from Character.codePointOf
        return IGNORE;
      } else {
        return extractOctal(value, i);
      }
    }

    private static EscapeSequence extractHexadecimal(String value, int i, int length) {
      if (i + 1 + length < value.length()) {
        try {
          int hexValue = Integer.parseInt(value.substring(i + 2, i + 2 + length), HEXADECIMAL_RADIX);
          return new EscapeSequence(2 + length, String.valueOf((char) hexValue));
        } catch (NumberFormatException ex) {
          return IGNORE;
        }
      } else {
        return IGNORE;
      }
    }

    private static EscapeSequence extractOctal(String value, int i) {
      // octal
      int octal = 0;
      int octalStart = (value.charAt(i + 1) == 'o') ? (i + 2) : (i + 1);
      int len = 0;
      int j = octalStart;
      while (len < 3 && j < value.length() && value.charAt(j) >= '0' && value.charAt(j) <= '7') {
        octal = octal * 8 + (value.charAt(j) - '0');
        j++;
        len++;
      }
      if (len > 0) {
        return new EscapeSequence(j - i, String.valueOf((char) octal));
      } else {
        return IGNORE;
      }
    }

  }

}
