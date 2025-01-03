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
package org.sonar.python.lexer;

import com.google.common.collect.ImmutableSet;
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;

import static com.sonar.sslr.test.lexer.LexerMatchers.hasComment;
import static com.sonar.sslr.test.lexer.LexerMatchers.hasToken;
import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.allOf;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PythonLexerTest {

  private static TestLexer lexer;

  @BeforeAll
  static void init() {
    lexer = new TestLexer();
  }

  private static class TestLexer {
    private LexerState lexerState = new LexerState();
    private Lexer lexer = PythonLexer.create(lexerState);

    List<Token> lex(String code) {
      lexerState.reset();
      return lexer.lex(code);
    }
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#comments
   */
  @Test
  void comments() {
    assertTrue(hasComment("# My comment ").matches(lexer.lex("# My comment \n new line")));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#string-literals
   */
  @Test
  void shortstring_literals() {
    assertTrue(hasToken("''", PythonTokenType.STRING).matches(lexer.lex("''")), "empty");
    assertTrue(hasToken("\"\"", PythonTokenType.STRING).matches(lexer.lex("\"\"")), "empty");

    assertTrue(hasToken("'hello world'", PythonTokenType.STRING).matches(lexer.lex("'hello world'")));
    assertTrue(hasToken("r'hello world'", PythonTokenType.STRING).matches(lexer.lex("r'hello world'")), "stringprefix");
    assertTrue(hasToken("R'hello world'", PythonTokenType.STRING).matches(lexer.lex("R'hello world'")), "stringprefix");
    assertTrue(hasToken("\"hello world\"", PythonTokenType.STRING).matches(lexer.lex("\"hello world\"")));
    assertTrue(hasToken("r\"hello world\"", PythonTokenType.STRING).matches(lexer.lex("r\"hello world\"")), "stringprefix");
    assertTrue(hasToken("R\"hello world\"", PythonTokenType.STRING).matches(lexer.lex("R\"hello world\"")), "stringprefix");

    assertTrue(hasToken("u'hello world'", PythonTokenType.STRING).matches(lexer.lex("u'hello world'")), "2.7.3 stringprefix");
    assertTrue(hasToken("ur'hello world'", PythonTokenType.STRING).matches(lexer.lex("ur'hello world'")), "2.7.3 stringprefix");

    assertTrue(hasToken("'\\''", PythonTokenType.STRING).matches(lexer.lex("'\\''")), "escaped single quote");
    assertTrue(hasToken("\"\\\"\"", PythonTokenType.STRING).matches(lexer.lex("\"\\\"\"")), "escaped double quote");

    assertTrue(hasToken("'", GenericTokenType.UNKNOWN_CHAR).matches(lexer.lex("'")), "unterminated");
    assertTrue(hasToken("\"", GenericTokenType.UNKNOWN_CHAR).matches(lexer.lex("\"")), "unterminated");
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#string-literals
   */
  @Test
  void longstring_literals() {
    assertTrue(hasToken("''''''", PythonTokenType.STRING).matches(lexer.lex("''''''")), "empty");
    assertTrue(hasToken("\"\"\"\"\"\"", PythonTokenType.STRING).matches(lexer.lex("\"\"\"\"\"\"")), "empty");

    assertTrue(hasToken("'''\n'''", PythonTokenType.STRING).matches(lexer.lex("'''\n'''")), "multiline");
    assertTrue(hasToken("\"\"\"\n\"\"\"", PythonTokenType.STRING).matches(lexer.lex("\"\"\"\n\"\"\"")), "multiline");

    assertTrue(hasToken("r'''\n'''", PythonTokenType.STRING).matches(lexer.lex("r'''\n'''")), "stringprefix");
    assertTrue(hasToken("r\"\"\"\n\"\"\"", PythonTokenType.STRING).matches(lexer.lex("r\"\"\"\n\"\"\"")), "stringprefix");

    assertTrue(hasToken("u'''\n'''", PythonTokenType.STRING).matches(lexer.lex("u'''\n'''")), "2.7.3 stringprefix");
    assertTrue(hasToken("ur'''\n'''", PythonTokenType.STRING).matches(lexer.lex("ur'''\n'''")), "2.7.3 stringprefix");
    assertTrue(hasToken("u\"\"\"\n\"\"\"", PythonTokenType.STRING).matches(lexer.lex("u\"\"\"\n\"\"\"")), "2.7.3 stringprefix");
    assertTrue(hasToken("ur\"\"\"\n\"\"\"", PythonTokenType.STRING).matches(lexer.lex("ur\"\"\"\n\"\"\"")), "2.7.3 stringprefix");

    assertTrue(hasToken("'''\\''''", PythonTokenType.STRING).matches(lexer.lex("'''\\''''")), "escaped single quote");
    assertTrue(hasToken("\"\"\"\\\"\"\"\"", PythonTokenType.STRING).matches(lexer.lex("\"\"\"\\\"\"\"\"")), "escaped double quote");

    assertTrue(hasToken("'", GenericTokenType.UNKNOWN_CHAR).matches(lexer.lex("'''")), "unterminated");
    assertTrue(hasToken("\"", GenericTokenType.UNKNOWN_CHAR).matches(lexer.lex("\"\"\"")), "unterminated");
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#string-literals
   */
  @Test
  void bytes_literal() {
    assertTrue(hasToken("br'hello world'", PythonTokenType.STRING).matches(lexer.lex("br'hello world'")));
    assertTrue(hasToken("rb'hello world'", PythonTokenType.STRING).matches(lexer.lex("rb'hello world'")));
    assertTrue(hasToken("br\"hello world\"", PythonTokenType.STRING).matches(lexer.lex("br\"hello world\"")));
    assertTrue(hasToken("rb\"hello world\"", PythonTokenType.STRING).matches(lexer.lex("rb\"hello world\"")));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#string-literals
   */
  @Test
  void longbytes_literal() {
    assertTrue(hasToken("b'''\n'''", PythonTokenType.STRING).matches(lexer.lex("b'''\n'''")));
    assertTrue(hasToken("b\"\"\"\n\"\"\"", PythonTokenType.STRING).matches(lexer.lex("b\"\"\"\n\"\"\"")));

    assertTrue(hasToken("br'''\n'''", PythonTokenType.STRING).matches(lexer.lex("br'''\n'''")));
    assertTrue(hasToken("rb'''\n'''", PythonTokenType.STRING).matches(lexer.lex("rb'''\n'''")));
    assertTrue(hasToken("br\"\"\"\n\"\"\"", PythonTokenType.STRING).matches(lexer.lex("br\"\"\"\n\"\"\"")));
    assertTrue(hasToken("rb\"\"\"\n\"\"\"", PythonTokenType.STRING).matches(lexer.lex("rb\"\"\"\n\"\"\"")));
  }

  /**
   * https://docs.python.org/3.6/reference/lexical_analysis.html#formatted-string-literals
   */
  @Test
  void formatted_string_literal() {
    Set<String> fstringPrefixes = ImmutableSet.of(
      "F",
      "f",
      "fr",
      "Fr",
      "fR",
      "FR",
      "rf",
      "rF",
      "Rf",
      "RF");
    for (String formattedStringLiteral : fstringPrefixes) {
      assertTrue(allOf(
        hasToken(formattedStringLiteral + "'", PythonTokenType.FSTRING_START),
        hasToken("'", PythonTokenType.FSTRING_END)).matches(lexer.lex(formattedStringLiteral + "''")));
    }
  }

  /**
   * https://docs.python.org/3.12/reference/lexical_analysis.html#formatted-string-literals
   */
  @Test
  void fstring_empty() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"\"")));
  }

  @Test
  void fstring_no_code() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken(" te st ", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\" te st \"")));
  }

  @Test
  void fstring_code_only() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("b", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"{a + b}\"")));
  }

  @Test
  void fstring() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("test ", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("b", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken(" foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"test {a + b} foo\"")));
  }

  @Test
  void fstring_multiple_code() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken(" + ", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("b", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"{a } + { b }\"")));
  }

  @Test
  void fstring_dict_access() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("mydict", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"{mydict[\"a\"]}\"")));
  }

  @Test
  void fstring_single_quote() {
    assertTrue(allOf(
      hasToken("f'", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken(" foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("'", PythonTokenType.FSTRING_END)).matches(lexer.lex("f'{a} foo'")));
  }

  @Test
  void fstring_triple_single_quote() {
    assertTrue(allOf(
      hasToken("f'''", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken(" foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("'''", PythonTokenType.FSTRING_END)).matches(lexer.lex("f'''{a} foo'''")));
  }

  @Test
  void fstring_triple_double_quote() {
    assertTrue(allOf(
      hasToken("f\"\"\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken(" foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"\"\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"\"\"{a} foo\"\"\"")));
  }

  @Test
  void fstring_with_escaped_quotes() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("\\\"", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\\\" foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"\\\"{a}\\\" foo\"")));
  }

  @Test
  void fstring_with_escaped_unicode() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("\\N{RIGHTWARDS ARROW} foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"\\N{RIGHTWARDS ARROW} foo\"")));
  }

  @Test
  void fstring_with_incorrect_unicode() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("RIGHTWARDS", GenericTokenType.IDENTIFIER),
      hasToken("ARROW", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken(" foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"\\N {RIGHTWARDS ARROW} foo\"")));
  }

  // Lambdas and walrus operators should be surrounded by parenthesis in an FString
  @Test
  void fstring_incorrect_lambda_format_specifier() {
    assertTrue(allOf(
      hasToken("f'", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("lambda"),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken(":", PythonPunctuator.COLON),
      hasToken(" a+42", PythonTokenType.FSTRING_MIDDLE),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("'", PythonTokenType.FSTRING_END)).matches(lexer.lex("f'{lambda a: a+42}'")));
  }

  @Test
  void fstring_incorrect_walrus_operator() {
    assertTrue(allOf(
      hasToken("f'", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken(":", PythonPunctuator.COLON),
      hasToken("=42", PythonTokenType.FSTRING_MIDDLE),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("'", PythonTokenType.FSTRING_END)).matches(lexer.lex("f'{a:=42}'")));
  }

  @Test
  void fstring_lambda() {
    assertTrue(allOf(
      hasToken("f'", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("(", PythonPunctuator.LPARENTHESIS),
      hasToken("lambda"),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken(":", PythonPunctuator.COLON),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("42", PythonTokenType.NUMBER),
      hasToken(")", PythonPunctuator.RPARENTHESIS),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("'", PythonTokenType.FSTRING_END)).matches(lexer.lex("f'{(lambda a: a+42)}'")));
  }

  @Test
  void fstring_walrus_operator() {
    assertTrue(allOf(
      hasToken("f'", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("(", PythonPunctuator.LPARENTHESIS),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken(":=", PythonPunctuator.WALRUS_OPERATOR),
      hasToken("42", PythonTokenType.NUMBER),
      hasToken(")", PythonPunctuator.RPARENTHESIS),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("'", PythonTokenType.FSTRING_END)).matches(lexer.lex("f'{(a:=42)}'")));
  }

  @Test
  void fstring_nested() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("1", PythonTokenType.NUMBER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("1", PythonTokenType.NUMBER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"{f\"{1+1}\"}\"")));
  }

  @Test
  void fstring_nested_mixed_number_of_quotes() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("f\"\"\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("1", PythonTokenType.NUMBER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("1", PythonTokenType.NUMBER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"\"\"", PythonTokenType.FSTRING_END),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"{f\"\"\"{1+1}\"\"\"}\"")));
  }

  @Test
  void fstring_nested_different_quotes() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("f'", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("1", PythonTokenType.NUMBER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("1", PythonTokenType.NUMBER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("'", PythonTokenType.FSTRING_END),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"{f'{1+1}'}\"")));
  }

  @Test
  void fstring_with_comment() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("abc", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasComment("# comment }\""),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("3", PythonTokenType.NUMBER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"abc{a # comment }\"\n + 3}\"")));
  }

  @Test
  void fstring_with_escaped_braces() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("abc{{a}} ", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("b", GenericTokenType.IDENTIFIER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("3", PythonTokenType.NUMBER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"abc{{a}} { b + 3}\"")));
  }

  @Test
  void fstring_with_newline_removed() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("abc{{a}} ", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("b", GenericTokenType.IDENTIFIER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("3", PythonTokenType.NUMBER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"abc{{a}} { b + 3}\"")));
  }

  @Test
  void fstring_with_dict_generator() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("for"),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("in"),
      hasToken("[", PythonPunctuator.LBRACKET),
      hasToken("1", PythonTokenType.NUMBER),
      hasToken(",", PythonPunctuator.COMMA),
      hasToken("2", PythonTokenType.NUMBER),
      hasToken("]", PythonPunctuator.RBRACKET),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"{ {a for a in [1,2]} }\"")));
  }

  @Test
  void fstring_format_specifier() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("abc ", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("b", GenericTokenType.IDENTIFIER),
      hasToken(":", PythonPunctuator.COLON),
      hasToken(".3f", PythonTokenType.FSTRING_MIDDLE),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"abc {a + b:.3f}\"")));
  }

  @Test
  void fstring_nested_fields_format_specifier() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("abc ", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("b", GenericTokenType.IDENTIFIER),
      hasToken(":", PythonPunctuator.COLON),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("width", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken(".", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("length", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"abc {a + b:{width}.{length}}\"")));
  }

  @Test
  void fstring_date_format_specifier() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("date", GenericTokenType.IDENTIFIER),
      hasToken(":", PythonPunctuator.COLON),
      hasToken("%B %d, %Y", PythonTokenType.FSTRING_MIDDLE),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"{date:%B %d, %Y}\"")));
  }

  @Test
  void fstring_complex_format_specifier() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("line", GenericTokenType.IDENTIFIER),
      hasToken("=", PythonPunctuator.ASSIGN),
      hasToken("!", GenericTokenType.UNKNOWN_CHAR),
      hasToken("r", GenericTokenType.IDENTIFIER),
      hasToken(":", PythonPunctuator.COLON),
      hasToken("20", PythonTokenType.FSTRING_MIDDLE),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"{line = !r:20}\"")));
  }

  @Test
  void fstring_escaped_regex_pattern() {
    assertTrue(allOf(
      hasToken("rf\"", PythonTokenType.FSTRING_START),
      hasToken("\\{{\\n\\}}\\\"", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\\\"", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("rf\"\\{{\\n\\}}\\\"{a}\\\"\"")));
  }

  @Test
  void fstring_double_backslash() {
    assertTrue(allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\\\\", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)).matches(lexer.lex("f\"{a}\\\\\"")));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#integer-and-long-integer-literals
   */
  @Test
  void integer_literals() {
    assertTrue(hasToken("0", PythonTokenType.NUMBER).matches(lexer.lex("0")));
    assertTrue(hasToken("00_000000_0", PythonTokenType.NUMBER).matches(lexer.lex("00_000000_0")));
    assertTrue(hasToken("7", PythonTokenType.NUMBER).matches(lexer.lex("7")));
    assertTrue(hasToken("7_2", PythonTokenType.NUMBER).matches(lexer.lex("7_2")));
    assertTrue(hasToken("0o177", PythonTokenType.NUMBER).matches(lexer.lex("0o177")));
    assertTrue(hasToken("0o177_22", PythonTokenType.NUMBER).matches(lexer.lex("0o177_22")));
    assertTrue(hasToken("0b100110111", PythonTokenType.NUMBER).matches(lexer.lex("0b100110111")));
    assertTrue(hasToken("0b_1001101_11", PythonTokenType.NUMBER).matches(lexer.lex("0b_1001101_11")));
    assertTrue(hasToken("0xdeadbeef", PythonTokenType.NUMBER).matches(lexer.lex("0xdeadbeef")));
    assertTrue(hasToken("0xdead_beef", PythonTokenType.NUMBER).matches(lexer.lex("0xdead_beef")));

    assertTrue(hasToken("9L", PythonTokenType.NUMBER).matches(lexer.lex("9L")), "2.7.3 long decimal integer");
    assertTrue(hasToken("0x77L", PythonTokenType.NUMBER).matches(lexer.lex("0x77L")), "2.7.3 long octal integer");
    assertTrue(hasToken("0b11L", PythonTokenType.NUMBER).matches(lexer.lex("0b11L")), "2.7.3 long binary integer");
    assertTrue(hasToken("0xffL", PythonTokenType.NUMBER).matches(lexer.lex("0xffL")), "2.7.3 long hex integer");

    assertTrue(hasToken("0700", PythonTokenType.NUMBER).matches(lexer.lex("0700")), "2.7.3 octal integer");
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#floating-point-literals
   */
  @Test
  void floating_point_literals() {
    assertTrue(hasToken("3.14", PythonTokenType.NUMBER).matches(lexer.lex("3.14")));
    assertTrue(hasToken("3_0.1_4", PythonTokenType.NUMBER).matches(lexer.lex("3_0.1_4")));
    assertTrue(hasToken("10.", PythonTokenType.NUMBER).matches(lexer.lex("10.")));
    assertTrue(hasToken("10.", PythonTokenType.NUMBER).matches(lexer.lex("10._")));
    assertTrue(hasToken(".001", PythonTokenType.NUMBER).matches(lexer.lex(".001")));
    assertTrue(hasToken("1e100", PythonTokenType.NUMBER).matches(lexer.lex("1e100")));
    assertTrue(hasToken("3.14e-10", PythonTokenType.NUMBER).matches(lexer.lex("3.14e-10")));
    assertTrue(hasToken("3_0.1_4e-1_0", PythonTokenType.NUMBER).matches(lexer.lex("3_0.1_4e-1_0")));
    assertTrue(hasToken("0e0", PythonTokenType.NUMBER).matches(lexer.lex("0e0")));
    assertTrue(hasToken("0_0e0_0", PythonTokenType.NUMBER).matches(lexer.lex("0_0e0_0")));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#imaginary-literals
   */
  @Test
  void imaginary_literals() {
    assertTrue(hasToken("3.14j", PythonTokenType.NUMBER).matches(lexer.lex("3.14j")));
    assertTrue(hasToken("10.j", PythonTokenType.NUMBER).matches(lexer.lex("10.j")));
    assertTrue(hasToken("10j", PythonTokenType.NUMBER).matches(lexer.lex("10j")));
    assertTrue(hasToken(".001j", PythonTokenType.NUMBER).matches(lexer.lex(".001j")));
    assertTrue(hasToken("1e100j", PythonTokenType.NUMBER).matches(lexer.lex("1e100j")));
    assertTrue(hasToken("10_2e1_00j", PythonTokenType.NUMBER).matches(lexer.lex("10_2e1_00j")));
    assertTrue(hasToken("3.14e-10j", PythonTokenType.NUMBER).matches(lexer.lex("3.14e-10j")));
    assertTrue(hasToken("3_0.1_400e-1_00j", PythonTokenType.NUMBER).matches(lexer.lex("3_0.1_400e-1_00j")));
    assertTrue(hasToken("10J", PythonTokenType.NUMBER).matches(lexer.lex("10J")), "uppercase suffix");
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#identifiers
   */
  @Test
  void identifiers_and_keywords() {
    assertTrue(hasToken("class", PythonKeyword.CLASS).matches(lexer.lex("class")));
    assertTrue(hasToken("identifier", GenericTokenType.IDENTIFIER).matches(lexer.lex("identifier")));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#operators
   * http://docs.python.org/reference/lexical_analysis.html#delimiters
   */
  @Test
  void operators_and_delimiters() {
    assertTrue(hasToken("<<", PythonPunctuator.LEFT_OP).matches(lexer.lex("<<")));
    assertTrue(hasToken("+=", PythonPunctuator.PLUS_ASSIGN).matches(lexer.lex("+=")));
    assertTrue(hasToken("@=", PythonPunctuator.MATRIX_MULT_ASSIGN).matches(lexer.lex("@=")));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#blank-lines
   */
  @Test
  void blank_lines() {
    assertThat(lexer.lex("    # comment\n")).hasSize(1);
    assertThat(lexer.lex("    \n")).hasSize(1);
    assertThat(lexer.lex("    ")).hasSize(1);
    assertThat(lexer.lex("line\n\n")).hasSize(3);
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#indentation
   */
  @Test
  void indent_dedent() {
    assertTrue(allOf(hasToken("    ", PythonTokenType.INDENT), hasToken("  ", PythonTokenType.DEDENT)).matches(lexer.lex("    STATEMENT\n  STATEMENT")));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#implicit-line-joining
   */
  @Test
  void implicit_line_joining() {
    assertTrue(not(hasToken("\n", PythonTokenType.NEWLINE)).matches(lexer.lex("month_names = ['January', \n 'December']")));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#explicit-line-joining
   */
  @Test
  void explicit_line_joining() {
    assertTrue(hasToken(PythonTokenType.NEWLINE).matches(lexer.lex("line\r\nline")));
    assertTrue(hasToken(PythonTokenType.NEWLINE).matches(lexer.lex("line\rline")));
    assertTrue(hasToken(PythonTokenType.NEWLINE).matches(lexer.lex("line\nline")));

    assertTrue(not(hasToken(PythonTokenType.NEWLINE)).matches(lexer.lex("line\\\r\nline")));
    assertTrue(not(hasToken(PythonTokenType.NEWLINE)).matches(lexer.lex("line\\\rline")));
    assertTrue(not(hasToken(PythonTokenType.NEWLINE)).matches(lexer.lex("line\\\nline")));
    assertTrue(hasToken(PythonTokenType.NEWLINE).matches(lexer.lex("line\\\n\nline")));
    assertTrue(not(hasToken(PythonTokenType.DEDENT)).matches(lexer.lex("  line\\\nline")));
    assertTrue(not(hasToken(PythonTokenType.DEDENT)).matches(lexer.lex("  line\\\r\n\rline")));
    assertTrue(not(hasToken(PythonTokenType.DEDENT)).matches(lexer.lex("  line\\\r\rline")));
    assertTrue(hasToken(PythonTokenType.DEDENT).matches(lexer.lex("  line\\\n\nline")));
    assertTrue(hasToken(PythonTokenType.DEDENT).matches(lexer.lex("  line\\\r\n\r\nline")));

    assertThat(lexer.lex("line\\\n    line")).hasSize(3);
  }

  @Test
  void mixed_tabs_spaces() {
    List<Token> tokens = lexer.lex("def fun():\n" +
      "   if True:\n" +
      "\tpass");
    assertThat(tokens.get(11).getType()).isEqualTo(PythonTokenType.INDENT);

    tokens = lexer.lex("def fun():\n" +
      "   if True:\n" +
      "  \tpass");
    assertThat(tokens.get(11).getType()).isEqualTo(PythonTokenType.INDENT);

  }

  @Test
  void non_ascii_characters() {
    assertTrue(hasToken(GenericTokenType.IDENTIFIER).matches(lexer.lex("_hello123")));
    assertTrue(hasToken(GenericTokenType.IDENTIFIER).matches(lexer.lex("こんにちは")));
    assertTrue(hasToken(GenericTokenType.IDENTIFIER).matches(lexer.lex("_你好")));
  }
}
