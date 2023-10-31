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
package org.sonar.python.lexer;

import static com.sonar.sslr.test.lexer.LexerMatchers.hasComment;
import static com.sonar.sslr.test.lexer.LexerMatchers.hasToken;
import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.allOf;
import static org.hamcrest.Matchers.not;
import static org.junit.Assert.assertThat;

import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;

import com.google.common.collect.ImmutableSet;
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;

class PythonLexerTest {

  private static TestLexer lexer;

  @BeforeAll
  static void init() {
    lexer = new TestLexer();
  }

  private static class TestLexer {
    private LexerState lexerState= new LexerState();
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
    assertThat(lexer.lex("# My comment \n new line"), hasComment("# My comment "));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#string-literals
   */
  @Test
  void shortstring_literals() {
    assertThat("empty", lexer.lex("''"), hasToken("''", PythonTokenType.STRING));
    assertThat("empty", lexer.lex("\"\""), hasToken("\"\"", PythonTokenType.STRING));

    assertThat(lexer.lex("'hello world'"), hasToken("'hello world'", PythonTokenType.STRING));
    assertThat("stringprefix", lexer.lex("r'hello world'"), hasToken("r'hello world'", PythonTokenType.STRING));
    assertThat("stringprefix", lexer.lex("R'hello world'"), hasToken("R'hello world'", PythonTokenType.STRING));
    assertThat(lexer.lex("\"hello world\""), hasToken("\"hello world\"", PythonTokenType.STRING));
    assertThat("stringprefix", lexer.lex("r\"hello world\""), hasToken("r\"hello world\"", PythonTokenType.STRING));
    assertThat("stringprefix", lexer.lex("R\"hello world\""), hasToken("R\"hello world\"", PythonTokenType.STRING));

    assertThat("2.7.3 stringprefix", lexer.lex("u'hello world'"), hasToken("u'hello world'", PythonTokenType.STRING));
    assertThat("2.7.3 stringprefix", lexer.lex("ur'hello world'"), hasToken("ur'hello world'", PythonTokenType.STRING));

    assertThat("escaped single quote", lexer.lex("'\\''"), hasToken("'\\''", PythonTokenType.STRING));
    assertThat("escaped double quote", lexer.lex("\"\\\"\""), hasToken("\"\\\"\"", PythonTokenType.STRING));

    assertThat("unterminated", lexer.lex("'"), hasToken("'", GenericTokenType.UNKNOWN_CHAR));
    assertThat("unterminated", lexer.lex("\""), hasToken("\"", GenericTokenType.UNKNOWN_CHAR));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#string-literals
   */
  @Test
  void longstring_literals() {
    assertThat("empty", lexer.lex("''''''"), hasToken("''''''", PythonTokenType.STRING));
    assertThat("empty", lexer.lex("\"\"\"\"\"\""), hasToken("\"\"\"\"\"\"", PythonTokenType.STRING));

    assertThat("multiline", lexer.lex("'''\n'''"), hasToken("'''\n'''", PythonTokenType.STRING));
    assertThat("multiline", lexer.lex("\"\"\"\n\"\"\""), hasToken("\"\"\"\n\"\"\"", PythonTokenType.STRING));

    assertThat("stringprefix", lexer.lex("r'''\n'''"), hasToken("r'''\n'''", PythonTokenType.STRING));
    assertThat("stringprefix", lexer.lex("r\"\"\"\n\"\"\""), hasToken("r\"\"\"\n\"\"\"", PythonTokenType.STRING));

    assertThat("2.7.3 stringprefix", lexer.lex("u'''\n'''"), hasToken("u'''\n'''", PythonTokenType.STRING));
    assertThat("2.7.3 stringprefix", lexer.lex("ur'''\n'''"), hasToken("ur'''\n'''", PythonTokenType.STRING));
    assertThat("2.7.3 stringprefix", lexer.lex("u\"\"\"\n\"\"\""), hasToken("u\"\"\"\n\"\"\"", PythonTokenType.STRING));
    assertThat("2.7.3 stringprefix", lexer.lex("ur\"\"\"\n\"\"\""), hasToken("ur\"\"\"\n\"\"\"", PythonTokenType.STRING));

    assertThat("escaped single quote", lexer.lex("'''\\''''"), hasToken("'''\\''''", PythonTokenType.STRING));
    assertThat("escaped double quote", lexer.lex("\"\"\"\\\"\"\"\""), hasToken("\"\"\"\\\"\"\"\"", PythonTokenType.STRING));

    assertThat("unterminated", lexer.lex("'''"), hasToken("'", GenericTokenType.UNKNOWN_CHAR));
    assertThat("unterminated", lexer.lex("\"\"\""), hasToken("\"", GenericTokenType.UNKNOWN_CHAR));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#string-literals
   */
  @Test
  void bytes_literal() {
    assertThat(lexer.lex("br'hello world'"), hasToken("br'hello world'", PythonTokenType.STRING));
    assertThat(lexer.lex("rb'hello world'"), hasToken("rb'hello world'", PythonTokenType.STRING));
    assertThat(lexer.lex("br\"hello world\""), hasToken("br\"hello world\"", PythonTokenType.STRING));
    assertThat(lexer.lex("rb\"hello world\""), hasToken("rb\"hello world\"", PythonTokenType.STRING));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#string-literals
   */
  @Test
  void longbytes_literal() {
    assertThat(lexer.lex("b'''\n'''"), hasToken("b'''\n'''", PythonTokenType.STRING));
    assertThat(lexer.lex("b\"\"\"\n\"\"\""), hasToken("b\"\"\"\n\"\"\"", PythonTokenType.STRING));

    assertThat(lexer.lex("br'''\n'''"), hasToken("br'''\n'''", PythonTokenType.STRING));
    assertThat(lexer.lex("rb'''\n'''"), hasToken("rb'''\n'''", PythonTokenType.STRING));
    assertThat(lexer.lex("br\"\"\"\n\"\"\""), hasToken("br\"\"\"\n\"\"\"", PythonTokenType.STRING));
    assertThat(lexer.lex("rb\"\"\"\n\"\"\""), hasToken("rb\"\"\"\n\"\"\"", PythonTokenType.STRING));
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
      assertThat(lexer.lex(formattedStringLiteral + "''"), allOf(
        hasToken(formattedStringLiteral + "'", PythonTokenType.FSTRING_START),
        hasToken("'", PythonTokenType.FSTRING_END)));
    }
  }

  /**
   * https://docs.python.org/3.12/reference/lexical_analysis.html#formatted-string-literals
   */
  @Test
  void fstring_empty() {
    assertThat(lexer.lex("f\"\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_no_code() {
    assertThat(lexer.lex("f\" te st \""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken(" te st ", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_code_only() {
    assertThat(lexer.lex("f\"{a + b}\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("b", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring() {
    assertThat(lexer.lex("f\"test {a + b} foo\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("test ", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("b", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken(" foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_multiple_code() {
    assertThat(lexer.lex("f\"{a } + { b }\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken(" + ", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("b", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_dict_access() {
    assertThat(lexer.lex("f\"{mydict[\"a\"]}\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("mydict", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_single_quote() {
    assertThat(lexer.lex("f'{a} foo'"), allOf(
      hasToken("f'", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken(" foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("'", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_triple_single_quote() {
    assertThat(lexer.lex("f'''{a} foo'''"), allOf(
      hasToken("f'''", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken(" foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("'''", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_triple_double_quote() {
    assertThat(lexer.lex("f\"\"\"{a} foo\"\"\""), allOf(
      hasToken("f\"\"\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken(" foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"\"\"", PythonTokenType.FSTRING_END)));
  }


  @Test
  void fstring_with_escaped_quotes() {
    assertThat(lexer.lex("f\"\\\"{a}\\\" foo\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("\\\"", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\\\" foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_with_escaped_unicode() {
    assertThat(lexer.lex("f\"\\N{RIGHTWARDS ARROW} foo\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("\\N{RIGHTWARDS ARROW} foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_with_incorrect_unicode() {
    assertThat(lexer.lex("f\"\\N {RIGHTWARDS ARROW} foo\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("RIGHTWARDS", GenericTokenType.IDENTIFIER),
      hasToken("ARROW", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken(" foo", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  // Lambdas and walrus operators should be surrounded by parenthesis in an FString
  @Test
  void fstring_incorrect_lambda_format_specifier() {
    assertThat(lexer.lex("f'{lambda a: a+42}'"), allOf(
      hasToken("f'", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("lambda"),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken(":", PythonPunctuator.COLON),
      hasToken(" a+42", PythonTokenType.FSTRING_MIDDLE),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("'", PythonTokenType.FSTRING_END)));
  }


  @Test
  void fstring_incorrect_walrus_operator() {
    assertThat(lexer.lex("f'{a:=42}'"), allOf(
      hasToken("f'", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken(":", PythonPunctuator.COLON),
      hasToken("=42", PythonTokenType.FSTRING_MIDDLE),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("'", PythonTokenType.FSTRING_END)));
  }
  @Test
  void fstring_lambda() {
    assertThat(lexer.lex("f'{(lambda a: a+42)}'"), allOf(
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
      hasToken("'", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_walrus_operator() {
    assertThat(lexer.lex("f'{(a:=42)}'"), allOf(
      hasToken("f'", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("(", PythonPunctuator.LPARENTHESIS),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken(":=", PythonPunctuator.WALRUS_OPERATOR),
      hasToken("42", PythonTokenType.NUMBER),
      hasToken(")", PythonPunctuator.RPARENTHESIS),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("'", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_nested() {
    assertThat(lexer.lex("f\"{f\"{1+1}\"}\""), allOf(
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
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_nested_mixed_number_of_quotes() {
    assertThat(lexer.lex("f\"{f\"\"\"{1+1}\"\"\"}\""), allOf(
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
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_nested_different_quotes() {
    assertThat(lexer.lex("f\"{f'{1+1}'}\""), allOf(
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
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_with_comment() {
    assertThat(lexer.lex("f\"abc{a # comment }\"\n + 3}\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("abc", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasComment("# comment }\""),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("3", PythonTokenType.NUMBER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_with_escaped_braces() {
    assertThat(lexer.lex("f\"abc{{a}} { b + 3}\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("abc{{a}} ", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("b", GenericTokenType.IDENTIFIER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("3", PythonTokenType.NUMBER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_with_newline_removed() {
    assertThat(lexer.lex("f\"abc{{a}} { b + 3}\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("abc{{a}} ", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("b", GenericTokenType.IDENTIFIER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("3", PythonTokenType.NUMBER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_with_dict_generator() {
    assertThat(lexer.lex("f\"{ {a for a in [1,2]} }\""), allOf(
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
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_format_specifier() {
    assertThat(lexer.lex("f\"abc {a + b:.3f}\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("abc ", PythonTokenType.FSTRING_MIDDLE),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("+", PythonPunctuator.PLUS),
      hasToken("b", GenericTokenType.IDENTIFIER),
      hasToken(":", PythonPunctuator.COLON),
      hasToken(".3f", PythonTokenType.FSTRING_MIDDLE),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_nested_fields_format_specifier() {
    assertThat(lexer.lex("f\"abc {a + b:{width}.{length}}\""), allOf(
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
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_date_format_specifier() {
    assertThat(lexer.lex("f\"{date:%B %d, %Y}\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("date", GenericTokenType.IDENTIFIER),
      hasToken(":", PythonPunctuator.COLON),
      hasToken("%B %d, %Y", PythonTokenType.FSTRING_MIDDLE),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_complex_format_specifier() {
    assertThat(lexer.lex("f\"{line = !r:20}\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("line", GenericTokenType.IDENTIFIER),
      hasToken("=", PythonPunctuator.ASSIGN),
      hasToken("!", GenericTokenType.UNKNOWN_CHAR),
      hasToken("r", GenericTokenType.IDENTIFIER),
      hasToken(":", PythonPunctuator.COLON),
      hasToken("20", PythonTokenType.FSTRING_MIDDLE),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_escaped_regex_pattern() {
    assertThat(lexer.lex("rf\"\\{{\\n\\}}\""), allOf(
      hasToken("rf\"", PythonTokenType.FSTRING_START),
      hasToken("\\{{\\n\\}}", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }

  @Test
  void fstring_double_backslash() {
    assertThat(lexer.lex("f\"{a}\\\\\""), allOf(
      hasToken("f\"", PythonTokenType.FSTRING_START),
      hasToken("{", PythonPunctuator.LCURLYBRACE),
      hasToken("a", GenericTokenType.IDENTIFIER),
      hasToken("}", PythonPunctuator.RCURLYBRACE),
      hasToken("\\\\", PythonTokenType.FSTRING_MIDDLE),
      hasToken("\"", PythonTokenType.FSTRING_END)));
  }
  /**
   * http://docs.python.org/reference/lexical_analysis.html#integer-and-long-integer-literals
   */
  @Test
  void integer_literals() {
    assertThat(lexer.lex("0"), hasToken("0", PythonTokenType.NUMBER));
    assertThat(lexer.lex("00_000000_0"), hasToken("00_000000_0", PythonTokenType.NUMBER));
    assertThat(lexer.lex("7"), hasToken("7", PythonTokenType.NUMBER));
    assertThat(lexer.lex("7_2"), hasToken("7_2", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0o177"), hasToken("0o177", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0o177_22"), hasToken("0o177_22", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0b100110111"), hasToken("0b100110111", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0b_1001101_11"), hasToken("0b_1001101_11", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0xdeadbeef"), hasToken("0xdeadbeef", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0xdead_beef"), hasToken("0xdead_beef", PythonTokenType.NUMBER));

    assertThat("2.7.3 long decimal integer", lexer.lex("9L"), hasToken("9L", PythonTokenType.NUMBER));
    assertThat("2.7.3 long octal integer", lexer.lex("0x77L"), hasToken("0x77L", PythonTokenType.NUMBER));
    assertThat("2.7.3 long binary integer", lexer.lex("0b11L"), hasToken("0b11L", PythonTokenType.NUMBER));
    assertThat("2.7.3 long hex integer", lexer.lex("0xffL"), hasToken("0xffL", PythonTokenType.NUMBER));

    assertThat("2.7.3 octal integer", lexer.lex("0700"), hasToken("0700", PythonTokenType.NUMBER));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#floating-point-literals
   */
  @Test
  void floating_point_literals() {
    assertThat(lexer.lex("3.14"), hasToken("3.14", PythonTokenType.NUMBER));
    assertThat(lexer.lex("3_0.1_4"), hasToken("3_0.1_4", PythonTokenType.NUMBER));
    assertThat(lexer.lex("10."), hasToken("10.", PythonTokenType.NUMBER));
    assertThat(lexer.lex("10._"), hasToken("10.", PythonTokenType.NUMBER));
    assertThat(lexer.lex(".001"), hasToken(".001", PythonTokenType.NUMBER));
    assertThat(lexer.lex("1e100"), hasToken("1e100", PythonTokenType.NUMBER));
    assertThat(lexer.lex("3.14e-10"), hasToken("3.14e-10", PythonTokenType.NUMBER));
    assertThat(lexer.lex("3_0.1_4e-1_0"), hasToken("3_0.1_4e-1_0", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0e0"), hasToken("0e0", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0_0e0_0"), hasToken("0_0e0_0", PythonTokenType.NUMBER));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#imaginary-literals
   */
  @Test
  void imaginary_literals() {
    assertThat(lexer.lex("3.14j"), hasToken("3.14j", PythonTokenType.NUMBER));
    assertThat(lexer.lex("10.j"), hasToken("10.j", PythonTokenType.NUMBER));
    assertThat(lexer.lex("10j"), hasToken("10j", PythonTokenType.NUMBER));
    assertThat(lexer.lex(".001j"), hasToken(".001j", PythonTokenType.NUMBER));
    assertThat(lexer.lex("1e100j"), hasToken("1e100j", PythonTokenType.NUMBER));
    assertThat(lexer.lex("10_2e1_00j"), hasToken("10_2e1_00j", PythonTokenType.NUMBER));
    assertThat(lexer.lex("3.14e-10j"), hasToken("3.14e-10j", PythonTokenType.NUMBER));
    assertThat(lexer.lex("3_0.1_400e-1_00j"), hasToken("3_0.1_400e-1_00j", PythonTokenType.NUMBER));
    assertThat("uppercase suffix", lexer.lex("10J"), hasToken("10J", PythonTokenType.NUMBER));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#identifiers
   */
  @Test
  void identifiers_and_keywords() {
    assertThat(lexer.lex("class"), hasToken("class", PythonKeyword.CLASS));
    assertThat(lexer.lex("identifier"), hasToken("identifier", GenericTokenType.IDENTIFIER));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#operators
   * http://docs.python.org/reference/lexical_analysis.html#delimiters
   */
  @Test
  void operators_and_delimiters() {
    assertThat(lexer.lex("<<"), hasToken("<<", PythonPunctuator.LEFT_OP));
    assertThat(lexer.lex("+="), hasToken("+=", PythonPunctuator.PLUS_ASSIGN));
    assertThat(lexer.lex("@="), hasToken("@=", PythonPunctuator.MATRIX_MULT_ASSIGN));
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
    assertThat(lexer.lex("    STATEMENT\n  STATEMENT"), allOf(hasToken("    ", PythonTokenType.INDENT), hasToken("  ", PythonTokenType.DEDENT)));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#implicit-line-joining
   */
  @Test
  void implicit_line_joining() {
    assertThat(lexer.lex("month_names = ['January', \n 'December']"), not(hasToken("\n", PythonTokenType.NEWLINE)));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#explicit-line-joining
   */
  @Test
  void explicit_line_joining() {
    assertThat(lexer.lex("line\r\nline"), hasToken(PythonTokenType.NEWLINE));
    assertThat(lexer.lex("line\rline"), hasToken(PythonTokenType.NEWLINE));
    assertThat(lexer.lex("line\nline"), hasToken(PythonTokenType.NEWLINE));

    assertThat(lexer.lex("line\\\r\nline"), not(hasToken(PythonTokenType.NEWLINE)));
    assertThat(lexer.lex("line\\\rline"), not(hasToken(PythonTokenType.NEWLINE)));
    assertThat(lexer.lex("line\\\nline"), not(hasToken(PythonTokenType.NEWLINE)));
    assertThat(lexer.lex("line\\\n\nline"), hasToken(PythonTokenType.NEWLINE));
    assertThat(lexer.lex("  line\\\nline"), not(hasToken(PythonTokenType.DEDENT)));
    assertThat(lexer.lex("  line\\\r\n\rline"), not(hasToken(PythonTokenType.DEDENT)));
    assertThat(lexer.lex("  line\\\r\rline"), not(hasToken(PythonTokenType.DEDENT)));
    assertThat(lexer.lex("  line\\\n\nline"), hasToken(PythonTokenType.DEDENT));
    assertThat(lexer.lex("  line\\\r\n\r\nline"), hasToken(PythonTokenType.DEDENT));

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
    assertThat(lexer.lex("_hello123"), hasToken(GenericTokenType.IDENTIFIER));
    assertThat(lexer.lex("こんにちは"), hasToken(GenericTokenType.IDENTIFIER));
    assertThat(lexer.lex("_你好"), hasToken(GenericTokenType.IDENTIFIER));
  }
}
