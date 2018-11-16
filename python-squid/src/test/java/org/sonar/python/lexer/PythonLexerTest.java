/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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

import com.google.common.base.Charsets;
import com.google.common.collect.ImmutableSet;
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;
import java.util.List;
import java.util.Set;
import org.junit.BeforeClass;
import org.junit.Test;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;

import static com.sonar.sslr.test.lexer.LexerMatchers.hasComment;
import static com.sonar.sslr.test.lexer.LexerMatchers.hasToken;
import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.allOf;
import static org.hamcrest.Matchers.not;
import static org.junit.Assert.assertThat;

public class PythonLexerTest {

  private static Lexer lexer;

  @BeforeClass
  public static void init() {
    lexer = PythonLexer.create(new PythonConfiguration(Charsets.UTF_8));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#comments
   */
  @Test
  public void comments() {
    assertThat(lexer.lex("# My comment \n new line"), hasComment("# My comment "));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#string-literals
   */
  @Test
  public void shortstring_literals() {
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
  public void longstring_literals() {
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
  public void bytes_literal() {
    assertThat(lexer.lex("br'hello world'"), hasToken("br'hello world'", PythonTokenType.STRING));
    assertThat(lexer.lex("rb'hello world'"), hasToken("rb'hello world'", PythonTokenType.STRING));
    assertThat(lexer.lex("br\"hello world\""), hasToken("br\"hello world\"", PythonTokenType.STRING));
    assertThat(lexer.lex("rb\"hello world\""), hasToken("rb\"hello world\"", PythonTokenType.STRING));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#string-literals
   */
  @Test
  public void longbytes_literal() {
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
  public void formatted_string_literal() {
    Set<String> formattedStringLiterals = ImmutableSet.of(
      "F'foo'",
      "f\"foo\"",
      "f'foo{name}'",
      "fr'foo'",
      "Fr'foo'",
      "fR'foo'",
      "FR'foo'",
      "rf'foo'",
      "rF'foo'",
      "Rf'foo'",
      "RF'foo'",
      "RF'foo\\n'",

      "F'''foo'''",
      "rF'''foo'''",
      "fR\"\"\"foo\"\"\""
      );
    for (String formattedStringLiteral : formattedStringLiterals) {
      assertThat(lexer.lex(formattedStringLiteral), hasToken(formattedStringLiteral, PythonTokenType.STRING));
    }
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#integer-and-long-integer-literals
   */
  @Test
  public void integer_literals() {
    assertThat(lexer.lex("7"), hasToken("7", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0o177"), hasToken("0o177", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0b100110111"), hasToken("0b100110111", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0xdeadbeef"), hasToken("0xdeadbeef", PythonTokenType.NUMBER));

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
  public void floating_point_literals() {
    assertThat(lexer.lex("3.14"), hasToken("3.14", PythonTokenType.NUMBER));
    assertThat(lexer.lex("10."), hasToken("10.", PythonTokenType.NUMBER));
    assertThat(lexer.lex(".001"), hasToken(".001", PythonTokenType.NUMBER));
    assertThat(lexer.lex("1e100"), hasToken("1e100", PythonTokenType.NUMBER));
    assertThat(lexer.lex("3.14e-10"), hasToken("3.14e-10", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0e0"), hasToken("0e0", PythonTokenType.NUMBER));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#imaginary-literals
   */
  @Test
  public void imaginary_literals() {
    assertThat(lexer.lex("3.14j"), hasToken("3.14j", PythonTokenType.NUMBER));
    assertThat(lexer.lex("10.j"), hasToken("10.j", PythonTokenType.NUMBER));
    assertThat(lexer.lex("10j"), hasToken("10j", PythonTokenType.NUMBER));
    assertThat(lexer.lex(".001j"), hasToken(".001j", PythonTokenType.NUMBER));
    assertThat(lexer.lex("1e100j"), hasToken("1e100j", PythonTokenType.NUMBER));
    assertThat(lexer.lex("3.14e-10j"), hasToken("3.14e-10j", PythonTokenType.NUMBER));
    assertThat("uppercase suffix", lexer.lex("10J"), hasToken("10J", PythonTokenType.NUMBER));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#identifiers
   */
  @Test
  public void identifiers_and_keywords() {
    assertThat(lexer.lex("class"), hasToken("class", PythonKeyword.CLASS));
    assertThat(lexer.lex("identifier"), hasToken("identifier", GenericTokenType.IDENTIFIER));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#operators
   * http://docs.python.org/reference/lexical_analysis.html#delimiters
   */
  @Test
  public void operators_and_delimiters() {
    assertThat(lexer.lex("<<"), hasToken("<<", PythonPunctuator.LEFT_OP));
    assertThat(lexer.lex("+="), hasToken("+=", PythonPunctuator.PLUS_ASSIGN));
    assertThat(lexer.lex("@="), hasToken("@=", PythonPunctuator.MATRIX_MULT_ASSIGN));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#blank-lines
   */
  @Test
  public void blank_lines() {
    assertThat(lexer.lex("    # comment\n")).hasSize(1);
    assertThat(lexer.lex("    \n")).hasSize(1);
    assertThat(lexer.lex("    ")).hasSize(1);
    assertThat(lexer.lex("line\n\n")).hasSize(3);
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#indentation
   */
  @Test
  public void indent_dedent() {
    assertThat(lexer.lex("    STATEMENT\n  STATEMENT"), allOf(hasToken("    ", PythonTokenType.INDENT), hasToken("  ", PythonTokenType.DEDENT)));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#implicit-line-joining
   */
  @Test
  public void implicit_line_joining() {
    assertThat(lexer.lex("month_names = ['January', \n 'December']"), not(hasToken("\n", PythonTokenType.NEWLINE)));
  }

  /**
   * http://docs.python.org/reference/lexical_analysis.html#explicit-line-joining
   */
  @Test
  public void explicit_line_joining() {
    assertThat(lexer.lex("line\r\nline"), hasToken(PythonTokenType.NEWLINE));
    assertThat(lexer.lex("line\rline"), hasToken(PythonTokenType.NEWLINE));
    assertThat(lexer.lex("line\nline"), hasToken(PythonTokenType.NEWLINE));

    assertThat(lexer.lex("line\\\r\nline"), not(hasToken(PythonTokenType.NEWLINE)));
    assertThat(lexer.lex("line\\\rline"), not(hasToken(PythonTokenType.NEWLINE)));
    assertThat(lexer.lex("line\\\nline"), not(hasToken(PythonTokenType.NEWLINE)));

    assertThat(lexer.lex("line\\\n    line")).hasSize(3);
  }

  @Test
  public void mixed_tabs_spaces() {
    List<Token> tokens = lexer.lex("def fun():\n" +
        "   if True:\n" +
        "\tpass");
    assertThat(tokens.get(11).getType()).isEqualTo(PythonTokenType.INDENT);

    tokens = lexer.lex("def fun():\n" +
        "   if True:\n" +
        "  \tpass");
    assertThat(tokens.get(11).getType()).isEqualTo(PythonTokenType.INDENT);


  }

}
